# torch 学习笔记

## view 和 reshape

view 和 reshape 都可以改变 tensor 的形状， view 是观察角度不同， 实际内存不变。 然而 reshape 也不能确保一定对副本进行形状变换。 所以如果想要对副本进行形状改变， 建议先 clone， 再 view

[[What's the difference between reshape and view in pytorch?](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch)](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch)

```python
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)
```



> 注：虽然`view`返回的`Tensor`与源`Tensor`是共享`data`的，但是依然是一个新的`Tensor`（因为`Tensor`除了包含`data`外还有一些其他属性），二者id（内存地址）并不一致。



## torch.nn

注意：`torch.nn`仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用`input.unsqueeze(0)`来添加一维。	





## [3.3.3 定义模型](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch?id=_333-定义模型)

在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。当模型结构变得更复杂时，这些步骤将变得更繁琐。其实，PyTorch提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。下面将介绍如何使用PyTorch更简洁地定义线性回归。

首先，导入`torch.nn`模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了`autograd`，而`nn`就是利用`autograd`来定义模型。`nn`的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承`nn.Module`，撰写自己的网络/层。一个`nn.Module`实例应该包含一些层以及返回输出的前向传播（forward）方法。下面先来看看如何用`nn.Module`实现一个线性回归模型。

```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构Copy to clipboardErrorCopied
```

输出：

```
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)Copy to clipboardErrorCopied
```

事实上我们还可以用`nn.Sequential`来更加方便地搭建网络，`Sequential`是一个有序的容器，网络层将按照在传入`Sequential`的顺序依次被添加到计算图中。

```python
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])Copy to clipboardErrorCopied
```

输出：

```
Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Linear(in_features=2, out_features=1, bias=True)
```



## [3.3.4 初始化模型参数](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch?id=_334-初始化模型参数)

在使用`net`前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。PyTorch在`init`模块中提供了多种参数初始化方法。这里的`init`是`initializer`的缩写形式。我们通过`init.normal_`将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。

```python
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)Copy to clipboardErrorCopied
```

> 注：如果这里的`net`是用3.3.3节一开始的代码自定义的，那么上面代码会报错，`net[0].weight`应改为`net.linear.weight`，`bias`亦然。因为`net[0]`这样根据下标访问子模块的写法只有当`net`是个`ModuleList`或者`Sequential`实例时才可以，详见4.1节。



## [3.4.5 交叉熵损失函数](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.4_softmax-regression?id=_345-交叉熵损失函数)

分类损失没法很好的评估分类错误程度。

MSE 太过严格， 且无法在一开始错误比较明显的时候， 提高loss， 提高梯度调整速度。

交叉熵比较好的解决了上述两个问题。

[交叉熵损失函数](https://zhuanlan.zhihu.com/p/35709485)