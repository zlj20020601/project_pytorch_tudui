import torch
from torch import nn
from torch.nn import Flatten, Linear, MaxPool2d, Conv2d, Sequential
from torch.utils.tensorboard import SummaryWriter


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

net = Net()
# print(net)
input = torch.ones(64,3,32,32)
output = net(input)
print(output.shape)

writer = SummaryWriter(log_dir="logs_seq")
writer.add_graph(net,input)
writer.close()
