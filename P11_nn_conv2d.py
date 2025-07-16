import  torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from test_tb import writer

dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

class Zlj(nn.Module):
    def __init__(self):
        super(Zlj,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x
zlj = Zlj()
# print(zlj)
writer = SummaryWriter("conv2d")
step = 0
for data in dataloader:
    imgs,targets = data
    output = zlj(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    # torch.Size([64, 6, 30, 30])
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step += 1

writer.close()