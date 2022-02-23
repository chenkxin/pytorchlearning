import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

# 制造数据
num_examples = 1000
num_inputs = 2
true_w = [3, 27]
true_b = 10

# torch.randn 是标准正太分布
featrues = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * featrues[:, 0] + true_w[1] * featrues[:, 1] + true_b
# 制造数据偏差
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# 读取数据
# 使用 data 包来读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(featrues, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

# 初始化模型参数
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
# b = torch.zeros(1, dtype=torch.float32)
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)

# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

# 利用 sequential
# 写法一
    # net = nn.Sequential(
    #     nn.Linear(num_inputs, 1)
    #     # 此处还可以添加其他层
    # )
# 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ...

print(net)
# print(net[0])
# 写法三
# from collections import OrderedDict
# net =

# 通过 net.parameters() 查看模型所有可学习参数
for param in net.parameters():
    print(param)

# 模型参数初始化
# class 方式定义网络的参数初始化
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) # 也可以直接修改bias的data: net.linear.bias.data.fill_(0)
# 当net是个ModuleList或者Sequential实例时
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零
        l.backward()
        optimizer.step()
    print("epoch %d   loss: %f" % (epoch, l.item()))

dense = net.linear
print(true_w, '\n', dense.weight)
print(true_b, '\n', dense.bias)