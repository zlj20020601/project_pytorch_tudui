import torch
import torchvision.datasets
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # self.conv2d1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2d2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv2d3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=10)
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)

        )

    def forward(self,x):
        # x = self.conv2d1(x)
        # x = self.maxpool1(x)
        # x = self.conv2d2(x)
        # x = self.maxpool2(x)
        # x = self.conv2d3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)#1024计算
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

loss_fn = torch.nn.CrossEntropyLoss()
net = Net()
for data in dataloader:
    inputs,targets = data
    outputs = net(inputs)
    loss = loss_fn(outputs,targets)
    loss.backward()
    print("ok")
    # print(loss.item())
#loss 是什么？
# 在 PyTorch 中，当你计算损失（例如 loss = loss_fn(outputs, targets)）时，loss 通常是一个 0 维的 PyTorch 张量（tensor）。#
# 尽管它只有一个数值，但它仍然是一个张量对象，而不是一个普通的 Python 数字。#
# 例如，如果你直接 print(loss)，你可能会看到类似 tensor(0.5123, grad_fn=<NllLossBackward0>) 这样的输出。
# 这里 grad_fn 表示这个张量是计算图的一部分，可以用于反向传播。
#
# 为什么需要 .item()？#
# 获取纯数值： .item() 方法是 PyTorch 张量的一个方法，用于将只包含一个元素的张量转换为标准的 Python 数字（例如 float 或 int）。#
# 用于打印或记录： 当你想要将损失值打印到控制台，或者将其存储到列表中用于绘图/分析时，通常需要一个纯粹的数字，而不是一个张量对象。.item() 就能满足这个需求。#
# 脱离计算图： item() 方法会把张量从计算图中分离出来。这意味着你不能再对这个 .item() 后的数值进行反向传播操作。
# 但对于损失值，通常我们只需要它的数值，而不是对其本身再进行反向传播，所以这不是问题。

