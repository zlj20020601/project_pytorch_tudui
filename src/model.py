import  torch.nn as nn
import  torch.nn.functional as F


class LeNet(nn.Module):
    # 定义网络层结构
    def __init__(self):
        super(LeNet,self).__init__()
        #super继承父类的构造函数 kernel_size卷积核的大小
        self.conv1 = nn.Conv2d(3,16,5)#彩色图片
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x= F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,32*5*5) #view中的一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
input1 = torch.randn([32,3,32,32])
model = LeNet()
print(model)
output = model(input1)