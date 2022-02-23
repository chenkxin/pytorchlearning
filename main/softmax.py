import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# mnist_train = torchvision.datasets.FashionMNIST(root='~/data/FasionMNIST', train=True, download=True, transform=transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root='~/data/FasionMNIST', train=False, download=True, transform=transforms.ToTensor())

# print(type(mnist_test))
# print(len(mnist_test), len(mnist_train))
# feature, label = mnist_train[0]
# print(feature.shape, label)

# 本函数已保存在d2lzh包中方便以后使用
# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]

# 下面定义一个可以在一行里画出多张图像和对应标签的函数。
# 本函数已保存在d2lzh包中方便以后使用
# def show_fashion_mnist(images, labels):
#     d2l.use_svg_display()
#     # 这里的_表示我们忽略（不使用）的变量
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.view((28, 28)).numpy())
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)
#     plt.show()

# 查看一下图片和标签
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

# 读取小批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# 获取和读取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, num_workers)

# 定义和初始化模型
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch_size, 1, 28, 28)
        y = self.linear(x.view(x[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)

我们将对x的形状转换的这个功能自定义一个FlattenLayer并记录在d2lzh_pytorch中方便后面使用。

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# class FlattenLayer(nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()
#     def forward(self, x): # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)

# from collections import OrderedDict

# net = nn.Sequential(
#     d2l.FlattenLayer(),
#     nn.Linear(num_inputs, num_outputs)
#     # OrderedDict([
#     #     ('flatten', FlattenLayer()),
#     #     ('linear', nn.Linear(num_inputs, num_outputs))
#     # ])
# )

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 训练模型
num_epochs = 5
d2l.train_ch3
