import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader  = DataLoader(dataset, batch_size=64, shuffle=True,drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()#继承父类
        self.linear1 = nn.Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output
net = Net()


for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    # a = torch.reshape(imgs,(1,1,1,-1))
    a = torch.flatten(imgs)
    print(a.shape)
    output = net(a)
    print(output.shape)

