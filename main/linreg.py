import torch
import numpy as np
import random

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
def data_iter(batch_size, features, lables):
    num_examples = len(features)
    indices = list(range(0, num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), lables.index_select(0, j)

batch_size = 10
for X, y in data_iter(batch_size, featrues, labels):
    print(X, y)
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数
def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 /2

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# 训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = square_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, featrues, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(featrues, w, b), labels)
    # train_l 是 tensor， 取 mean， 再 item， 使之成为标量
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)