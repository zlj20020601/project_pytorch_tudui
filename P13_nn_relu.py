import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([
#     [1,-0.5],
#     [-1,3]
# ])
# input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)
dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False,num_workers=0,drop_last=True)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.relu1 = nn.ReLU()
#
#     def forward(self,input):
#         output = self.relu1(input)
#         return output
# net = Net()
# output = net(input)
# print(output)
writer = SummaryWriter("logs_sigmoid")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output
net = Net()
step = 0
for data in dataloader:
    input,target = data
    writer.add_images("input",input,step)
    output = net(input)
    writer.add_images("output",output,step)
    step += 1

writer.close()