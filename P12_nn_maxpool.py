import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# PyTorch 的大部分神经网络层（包括卷积层、池化层、激活函数等）以及许多数学运算，主要设计用于处理浮点数类型的数据（如 torch.float32 或 torch.float64）。
# 当你使用 torch.tensor([ ... ]) 创建张量时，如果没有显式指定 dtype，PyTorch 会根据你提供的数据类型自动推断。
# 由于你提供的都是整数，它会默认为 torch.int64 (即 Long 类型)。
dataset = torchvision.datasets.CIFAR10("data",train=False,download=True,transform=transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True,num_workers=0,drop_last=True)
# input = torch.tensor([
#     [1,2,0,3,1],
#     [0,1,2,3,1],
#     [1,2,1,0,0],
#     [5,2,3,1,1],
#     [2,1,0,1,1]
# ], dtype=torch.float32) #显示指定dtype

# input= torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.maxpool1 = nn.MaxPool2d(3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

writer = SummaryWriter(log_dir="logs_maxpool")
step = 0
net = Net()
for data in dataloader:
    input,target = data
    writer.add_images("input",input,step)

    output = net(input)
    writer.add_images("output",output,step)
    step += 1

writer.close()
