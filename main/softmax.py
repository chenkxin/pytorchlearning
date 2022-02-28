import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, 4)

num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# class FlattenLayer(nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()
#     def forward(self, x): # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)

from collections import OrderedDict

# net = nn.Sequential(
#     # FlattenLayer(),
#     # nn.Linear(num_inputs, num_outputs)
#     OrderedDict([
#         ('flatten', FlattenLayer()),
#         ('linear', nn.Linear(num_inputs, num_outputs))
#     ])
# )

# 然后，我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)