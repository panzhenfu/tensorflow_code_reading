# 读TensorFlow 源码笔记(2): tensorflow的控制流算子(control_flow_op)

- 介绍专门为处理控制流而添加的五个TensorFlow原语运算符,
- 演示如何将高级控制流结构编译为包含这五个原语的数据流图

- 解释TensorFlow运行时如何执行这些数据流图，包括在一组混合设备（如CPU、GPU和TPU）上的分布式执行，并描述自动区分对控制流结构的作用。

## 控制流原语
在TensorFlow中，控制流的基本设计原则中引入一组非常小的简单、原始的运算符，这些运算符广泛应用在TensorFlow的复杂控制流中。我们希望这些原语具有灵活性和表现力，成为高级领域特定语言（DSL）的良好编译目标。它们应该很好地适应TensorFlow的数据流模型，并且应该能够并行和分布式执行以及自动区分。

有五个控制流原语运算符，如下所示。它们与Dennis和Arvind开发的数据流机器中引入的控制流原语非常相似。Switch和Merge的结合允许我们实现条件语句。所有五个原语一起允许我们实现while循环。
![tensorflow primitives](./control_flow_op/tensorflow_primitives.jpg)

在TensorFlow中，每个操作都在一个执行帧(execution frame)中执行，控制流原语负责创建和管理这些执行帧(execution frame)。直观地说，对于每个while循环，TensorFlow运行时设置一个执行帧(execution frame)，并在执行帧(execution frame)内运行属于while循环的所有操作(op)。执行帧(execution frame)可以嵌套。嵌套的while循环在嵌套的执行帧(execution frame)中运行。来自不同执行帧(execution frame)的操作(op)可以并行运行，只要它们之间没有数据依赖关系。

**Switch :** *switch*算子根据控制输入`p`的布尔张量(tensor of bool)将输入张量`d`(input tensor)转发到其输出之一。当*switch*的两个输入`p`和`d`都可用时，*switch*被启用以执行。

**Merge :** *merge*运算符将其一个可用输入转发到其输出。当*merge*的任何输入可用时，将启用该*merge*以执行。如果有多个可用输入，则不指定输出哪个可用输入。

**Enter(name) :** *Enter*运算符将其输入转发到由给定名称(name)唯一标识的执行帧(execution frame)。此Enter op用于将一个执行帧(execution frame)中的张量传递给子执行帧。同一子执行帧可以有多个*Enter*操作，每个*Enter*操作都使该子执行帧中的张量可用（异步）。当输入可用时，将启用*Enter*执行。当对该帧执行第一个*Enter*操作时，将在TensorFlow运行时实例化一个新的执行帧。

**Exit :** *Exit*运算符将值从执行帧转发到其父执行帧。此*Exit* op用于将子执行帧中计算的张量返回到其父帧。父帧可以有多个退出操作，每个都异步地将张量传递回父帧。退出在其输入可用时启用。

**NextIteration :** *NextIteration*运算符将其输入转发到当前执行帧中的下一个迭代。TensorFlow运行时在执行帧中跟踪迭代。在执行帧中执行的任何操作都有一个唯一的迭代id，它允许我们在迭代计算中唯一地标识同一操作的不同调用。注意，在一个执行帧中可以有多个NextIteration操作。TensorFlow运行时在迭代N执行第一个NextIteration操作时启动迭代N+1。随着更多的Tensor通过执行NextIteration操作进入迭代，该迭代中的更多操作将准备好执行。当输入可用时，将启用NextIteration。

## 控制流结构的编译
 通过添加以上这五个控制流原语，条件(cond)和循环(while_)等高级编程结构现在就可以编译成数据流图，这些数据流图可以由TensorFlow运行时执行。

 ### 条件(cond)运算符
 下面是构建cond(pred，fn1，fn2)数据流图的高级伪代码。为了简单起见，此中忽略了实际实现中的许多重要问题。读者可以在control_flow_ops.py中找到实现。
 ```python 
 # Build the graph for the true branch
context_t = CondContext(pred, branch=1)
res_t = context_t.Call(fn1)
# Build the graph for the false branch
context_f = CondContext(pred, branch=0)
res_f = context_f.Call(fn2)
# Add the Merge nodes for the outputs
merges = [Merge([f, t]) for (f, t) in zip(res_f, res_t)]
return merges
 ```

 对于cond的每个分支，我们为条件语句创建一个新的控制流上下文，并在上下文中调用其图形构造函数（fn1或fn2）。条件上下文允许捕获任何外部张量（不是在上下文中创建的）并插入适当的Switch op来保护其进入分支(branch)。这确保了分支中的任何操作都只能在执行该分支时执行。由于TensorFlow的异步执行模型，这些外部张量可能在非常不同的时间变得可用，因此还需要为每个外部张量使用一个switch操作以最大限度地提高并行性。

 每个分支返回一个张量列表作为结果（ref_t或res_f）；然后添加一个合并节点列表，分别合并(merge)每个输出的true和false值。同样，输出可以在非常不同的时间进行计算，因此我们对每个输出使用一个合并(merge)操作，这允许能够尽快启用下游计算。

 作为一个例子，看看这个简单的程序。
 ![cond_control_flow](./control_flow_op/cond_control_flow.jpg)

 ```python 
 tf.cond(x < y, lambda : tf.add(x, z), lambda : tf.square(y))
 ```
 在生成的数据流图中，在true/false分支上插入开关(switch)操作以控制张量x、y 和z 的流。由于add的输入来自switch ops的true输出，因此仅当x < y 为真时才执行add op。类似地，Square op仅在x < y 为false时执行。最后的Merge op发出Add或Square的结果。如果有多个输出，将有多个merge操作，每个输出一个结果。

 有多种方法可以使用Switch和Merge对cond进行编码。这里的编码主要是因为它使cond的自动区分变得更简单。

### while_循环运算符
下面是构建while_循环(pred，body，loop_vars)数据流图的高级伪代码：
```python
while_context = WhileContext()
while_context.Enter()
# Add the Enter nodes for each loop variable.
enter_vars = [Enter(x, frame_name) for x in loop_vars]
# Add the Merge nodes. Note that input[1] will be updated later.
merge_vars = [Merge([x, x]) for x in enter_vars]
# Build the loop pred subgraph.
pred_result = pred(*merge_vars)
# Add the Switch nodes.
switch_vars = [Switch(x, pred_result) for x in merge_vars]
# Build the loop body subgraph.
body_result = body(*[x[1] for x in switch_vars])
# Add the NextIteration nodes.
next_vars = [NextIteration(x) for x in body_result]
# Form the cycles for the loop.
for m, v in zip(merge_vars, next_vars):
m.op._update_input(1, v)
# Add the Exit nodes.
exit_vars = [Exit(x[0]) for x in switch_vars]
while_context.Exit()
return exit_vars
```

整个while循环图是在while循环的控制流上下文中创建的。这里的基本思想很简单。
从循环变量开始，我们为每个变量添加一个*Enter*操作，然后添加一个*Merge*操作。然后使用结果（merge_vars）构建pred子图，该子图计算循环终止条件。

在添加开关(switch)操作之后，我们使用*switch*的true输出为*while*循环的主体构建子图。循环体的结果需要进入下一个迭代，因此我们添加*next iteration*操作并将它们连接回*Merge*操作的第二个输入。这形成了循环，允许我们在执行图时多次重复运行同一个操作。

开关(switch)操作的false输出是整个while循环的输出，因此我们将*exit*操作添加到它们并返回*exit*操作的输出。与*cond*类似，*while*循环上下文用于跟踪*pred*和*body lambdas*中使用的外部张量。这些外部张量被视为循环常量，自动为每个这样的外部张量插入一个*Enter op*，使其在*while*循环上下文中可访问。嵌套循环需要添加嵌套的*Enter ops*。

为一个简单程序生成的图.
![while_control_flow](./control_flow_op/while_control_flow.jpg)

```python
tf.while_loop(lambda i : i< 10, lambda i : tf.add(i,1),[0])
```
对于这个例子，只有一个循环变量。如果有多个循环变量，我们将有多个Enter、Merge、Switch、NextIteration和Exit操作。这使得可以跨多个循环和循环内的多个迭代执行并行。

cond和while_循环的这种转换支持条件句和循环的任意嵌套。例如，循环体可以调用另一个while_循环，该循环将递归地转换为嵌套子图。转换确保每个循环静态地分配一个唯一的帧名(frame name)。

## 实现
要在多个设备上运行，TensorFlow会自动将操作(op)分配给设备集(device set)。根据设备的位置，TensorFlow自动将数据流图划分为一组子图(subgraph)，每个设备分配一个子图(subgraph)。当一条边被分区(partition)破坏时，会自动插入一对发送(send)和接收(recv)节点，用于跨设备传输张量(tensor)。一对send和recv使用一个唯一的密钥(key)进行通信，recv主动从send中拉去数据。例如，下面是将一个图分区到两个设备上的结果。TensorFlow对分区没有任何限制：只要可以在设备上进行计算，就可以将节点分配给该设备。
![tensorflow_partitions_send_recv](./control_flow_op/tensorflow_partitions_send_recv.jpg)

子图的执行由子图分配给的设备的本地执行器(executor)管理。执行器(executor)从源节点(source node)开始并重复执行就绪节点(ready
nodes)。当一个节点的所有输入都可用时，该节点(合并节点(merge node)除外)将准备就绪。注意，子图中的所有recv节点都被视为源节点(source node)。
在没有控制流的情况下，图的执行在概念上是非常简单的：每个节点只执行一次，执行是在所有节点都执行时完成的。控制流引入了相当多的复杂性。一个节点现在可以执行任意次数，包括0。执行器需要能够管理同一节点的多个实例（可能并发）的执行，并确定图形执行的完成。

为了跟踪执行期间生成的张量，执行器中的张量表示为一个元组d=(value，is_dead，tag)，其中value是实际的张量，is_dead是一个布尔值，指示张量是否在条件的未指定分支上，tag是唯一标识张量(以及生成张量的节点的执行实例)的字符串。直观地说，tag定义了一个执行上下文(executor context)，在一个执行上下文中，节点最多执行一次。tag是send/recv对的通信密钥(key)的一部分，用于区分同一send/recv对的多个调用。
执行器遵循以下执行规则（注意：节点的所有输入必须具有相同的标记(tag)）:
- Switch(p, d) = (r<sub>1</sub>, r<sub>2</sub> ) :

  r<sub>1</sub> = (value(d), p || is_dead(d), tag(d))

  r<sub>2</sub> = (value(d), !p || is_dead(d), tag(d))

- Merge(d<sub>1</sub> , d<sub>2</sub> ) = r :

  r = if is_dead(d<sub>1</sub> ) then d<sub>2</sub> else d<sub>1</sub>

-  Enter(d, frame_name) = r :

    value(r) = value(d)

    is_dead(r) = is_dead(d)

    tag(r) = tag(d)/frame_name/0

- Exit(d) = r :

  value(r) = value(d)

  is_dead(r) = is_dead(d)

  tag(r) = tag<sub>1</sub> where tag(d) = tag<sub>1</sub> /frame_name/n

- NextIteration(d) = d<sub>1</sub>:

  value(d<sub>1</sub> ) = value(d)

  is_dead(<sub>1</sub> ) = is_dead(d)

  tag(<sub>1</sub> ) = tag<sub>1</sub> /frame_name/(n+1) where tag(d) = tag<sub>1</sub> /frame_name/n

- Op(d<sub>1</sub> , …, d<sub>m</sub> ) = (r<sub>1</sub> , …, r<sub>n</sub> ) :

  value(r<sub>i</sub> ) = Op.Compute(value(d<sub>1</sub> ), …, value(d<sub>m</sub>)) if !is_dead(r<sub>i</sub>)

  is_dead(r<sub>i</sub> ) = any(is_dead(d<sub>1</sub> ), … is_dead(d<sub>m</sub> )), for all i

  tag(r<sub>i</sub> ) = tag(d<sub>1</sub> ), for all i

  最后一条规则适用于所有非控制流节点。请注意，实际计算仅在所有输入都未停止时执行。如果存在死区输入，我们将跳过计算并在下游传播死区信号。这种死区传播用于支持控制流的分布式执行。

  ## 分布式条件执行
  对于分布式执行，可以将cond分区到多个设备上，如下所示。
  ![distribution_cond_control_flow](./control_flow_op/distribution_cond_control_flow.jpg)

  ~~~<font size=4>由于任何recv节点都是源节点，并且可以无条件地启动，因此即使设备B上的recv位于cond的untaken分支上，它也可以启动。为了使untaken分支上的recv高兴，我们在从send到recv的设备之间传播is_dead标志。传播可以在任意数量的设备上继续。这个简单的传播方案处理嵌套条件的分布式执行，并与while循环的分布式执行进行良好的交互。 没看懂！！！！！！！！！！！！！！！</font>~~~
## 分布式While循环
对于分布式执行，一个while循环，特别是循环体，可以被划分到多个设备上。如果纯粹地应用跨设备边添加send/recv节点的分区方案，设备上的本地执行器将没有足够的信息来正确运行while循环。 
![naive_partitioning_breaks](./control_flow_op/naive_partioning_breaks.png)

让我们用一个简单的例子来说明这些问题。在上面的例子中，Op在循环体中，并被分配给设备B。一个简单的分区只需要使用一对send/recv节点就可以将边缘从Switch断开到Op。但是，这将不起作用，因为设备B不知道recv和Op节点是while循环的一部分，并且将在一次迭代后终止执行。解决方案是要重写数据流图，在每个分区中添加一个控制循环状态机（如下设备B的右下角所示）。标量张量0用作控制循环的输入节点。
![distributing_control_loop](./control_flow_op/distributing_control_loop.jpg)

这些控制循环提供足够的信息，允许设备上的执行器像以前一样独立运行，通过send/recv节点相互通信。注意虚线是控制边。

更详细地说，让我们首先看看while循环只运行0次迭代的基本情况：
- 在设备A上，执行从节点Enter、Merge、P和Switch开始。由于P为false，连接Switch的send会将死区(dead signal)信号传播到设备B，并且设备A上的Exit也会运行，从而启用循环外节点的并发执行。连接到P的Send会将布尔张量False发送到设备B。还触发执行Recv，等待设备B的返回值。
- 在设备B上，执行从节点Enter和Merge开始。执行Merge将启用两个recv。Switch的Recv将收到False，因此Next将得到一个死张量。下一步是停止死亡(dead)的传播。Op的Recv将得到一个死张量(dead tensor)，这样Op的Send将把一个死张量(dead tensor)发送回设备A。此时，设备B没有未完成的ops，因此执行终止。
- 回到设备A，Next的Recv得到一个死张量。下一次运行时，由于它停止了死区的传播，设备A没有未完成的操作，因此执行终止。

现在假设while循环运行一个或多个迭代：
- 在设备A上，由于P在第一次迭代时为true，因此会向设备B发送一个实张量。执行Recv，等待设备B的值。
- 在设备B上，控制回路状态机(control-loop state machine)运行并启用recv。Op的Recv从设备A得到一个实张量；Op被执行并且发送一个实张量回设备A。Switch的Recv得到布尔张量True。执行Next和Merge，进一步为下一次迭代启用recv。
- 回到设备A，Recv得到一个真正的张量。接下来，Merge和P被执行。根据P的值，将执行基本情况或新的迭代。

注意，在执行过程中有很多并行性。例如，设备B在接收到P的值后可以开始下一个迭代或退出。参与设备可以并行运行多个迭代，并且两个参与设备可以在同一循环的不同迭代上工作。

while循环的分布式执行的开销是，每个参与设备在每次迭代时都需要从产生P的设备接收布尔张量。考虑到执行过程中的并行性，应该在很大程度上隐藏开销。

下面显示了在跨多个设备分区while循环时数据流图的外观。控制循环被添加到每个分区，并控制while循环中的recv。重写后的图在语义上等价于原始图。
![graph_partition_rewrite](./control_flow_op/graph_partition_rewrite.jpg)

对于嵌套的while循环，只将控制循环堆栈如下。注意，如果一个设备只有外环的节点，不会为该设备上的任何内环添加控制环。
![nest_while_loop](./control_flow_op/nest_while_loop.png)

## 自动微分

TensorFlow支持自动微分。例如，用户可以定义一个具有损失函数的神经网络，TensorFlow将自动求导并构建反向传播数据流图。本节说明了TensorFlow如何在cond和while_循环存在时自动构建反向传播图。

反向传播算法通过反向遍历前向图中的ops，通过调用ops的梯度函数逐步构造梯度图。op的梯度函数定义了计算op的符号梯度的子图。梯度函数可以使用op的输入/输出值，因此在前向计算中产生的一些张量将保留一段时间，直到在backprop中使用为止。例如，下面显示了一个正向操作及其梯度图。G（Op）是Op的梯度子图，x和y的值将保存在内存中，直到G（Op）被执行。

![back_forward_progration](./control_flow_op/back_forward_progration.png)

一旦构建了整个数据流图，TensorFlow运行时将自动对该图进行分区，并将执行分布在多个设备上。因此，TensorFlow中的梯度计算也将分布到多个设备上运行。
直观地说，在我们的cond和while_循环的高层结构中，控制流操作符的反向传播只是按照以下方式反向流动：Exit的梯度是Enter；Switch的梯度是Merge（对于cond）或NextIteration，然后是Merge（对于while_循环）；Merge的梯度是Switch；关系的梯度是恒等的，梯度的Enter就是Exit。TensorFlow支持嵌套条件和while循环的反向传播。

### 有条件的反向传播
直观地讲，cond(p，fn1，fn2)的梯度是cond(p，g_fn1，g_fn2)，其中g_fn1和g_fn2分别是fn1和fn2的梯度。下面显示当cond未嵌套在while循环中时cond的基本反向传播。假设Op在cond的真分支上。嵌套在while循环中的cond需要更多的工作来记住前向循环每次迭代的p值。稍后再看一下while循环的backprop。
![cond_backforward](./control_flow_op/cond_backforward.jpg)

forward Merge被转换成一个Switch，它使用与forward Switch相同的谓词p。梯度g<sub>y</sub>分发到Switch的两个分支上。forward Switch变为了Merge。如果forward中只使用forward Switch的一个分支，需要添加一个零，如下所示，以确保始终有一个活的梯度流过backprop中的Merge。这个0 由一个Switch来控制，因此只有当p为false时，它才会被发送到Merge中。
![Switch2Merge](./control_flow_op/Switch2Merge.jpg)

### While循环的反向传播

直观地说，while_loop(pred，body)的梯度类似于一下的while loop形式：
```python
def pred(i, _): return i < N
while_loop(pred, g_body, [0] + g_vars)
```
其中N是forward while循环运行的迭代次数，g_body是forward循环体的梯度，g_vars是循环变量的初始值。稍后将看到，g_vars包含forward while循环变量的初始梯度。while循环及其backprop while循环的图形大致如下：
![while_backprop](./control_flow_op/while_backprop.jpg)

backprop循环由N控制，N 为前向循环将运行的迭代次数，其中假设循环条件pred是不可训练的。G（Body）是运算主体的梯度。

Body可能再次包含while循环，因此此构造可以递归地发生以处理嵌套while循环。
到目前为止，这一描述相当过于简单化。例如，N在图构造时是静态未知的。更重要的是，G（Body）可能使用由forward循环体生成的值，我们希望保留这些值，以避免在backprop中重新计算它们。TensorFlow中的解决方式是重写forward while循环的graph，以添加计算和(或)保存backprop中所需的值的逻辑。

为了计算N，我们将以下子图添加到forward while循环中。因此，N将由前向循环动态计算，并发送给backprop循环的循环次数计数器作为变量的初始值。
![while_forward_dynamic_fed_backprop](./control_flow_op/while_forward_dynamic_fed_backprop.jpg)

为了在backprop循环中重用前向的值，TensorFlow在backprop while循环的构造过程中自动检测backprop中所需的前向值。对于每个这样的前向值x，会自动引入一个堆栈，并在前向循环(forward while)中添加节点，以在每次迭代时将其值保存到堆栈中。backprop循环按相反的顺序使用堆栈中的值。堆栈位于forward和backprop循环之外，由这两个循环共享。
![while_forward_backprop_share_stack](./control_flow_op/while_forward_backprop_share_stack.jpg)

实际的图形构造实际上比这更微妙和复杂。tensorflow在这里还涉及很多细节，比如还有一下这些问题：
- 为了确保正确性，必须确保堆栈推送和pop按其各自循环的迭代顺序排列。还要确保先在前向循环中向栈中push值之后，才可能在堆栈中弹出到backprop，这需要使用控制边(control edge)强制执行排序。
- 为了提高性能，TensorFlow将堆栈推送和弹出操作设为异步操作，以便它们可以与实际计算并行运行。例如，op（甚至未来的迭代）可以与Push并行运行。
- 如果op位于while循环中嵌套的cond中，那么cond的谓词必须正确地确保push和pop操作正确执行。
- 如果值立即被backprop中的reduce op（例如Shape、Rank或Size）减少，TensorFlow会将reduce op移动到forward循环以减少内存使用。

对于循环变量，如前所述，反向传播的梯度的Enter是Exit，以上就是它所做的一切。对于循环常数，TensorFlow还添加一个子图来累积它们的梯度，如下所示。
![while_constant_gradient_accumulate](./control_flow_op/while_constant_gradient_accumulate.png)

假设x是向前的循环常数。在backprop中，每次迭代都会生成x的部分梯度。所以在backprop中添加小的累加子图来将所有这些部分梯度相加。出口处的最终g<sub>x</sub>是所有部分梯度的总和。注意，累积是迫不及待地完成的，以并行迭代次数为界。这与静态展开不同，在静态展开中，AddN的使用需要同时激活所有的局部梯度。
这种构造对嵌套条件和循环都有效。对于嵌套在while循环中的cond，TensorFlow引入一个堆栈来保存每次前向迭代时谓词的值，并在backprop中使用堆栈中的值（以相反的顺序）。对于嵌套循环，当遇到嵌套在循环体中的内部while循环时，将递归调用此构造。

一个重要的优化是内存交换。正如我们所看到的，对于backprop中需要的每个正向值v，将其在所有迭代v<sub>1</sub>，…，v<sub>N</sub>中的值保存在堆栈中，以便在backprop中重用它们。这可能是在内存有限的设备（如gpu）上进行培训的限制。我们使用内存交换将堆栈中存储的值从GPU异步移动到CPU，并在backprop中需要时将它们移回GPU内存。