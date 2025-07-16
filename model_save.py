import torch
import torchvision
from torch import nn
import torch.nn.functional as F

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1(模型结构+模型参数）
torch.save(vgg16,"vgg16_method1.pth")
# 保存方式2（模型参数，官方推荐） 保存状态字典是为了避免兼容性问题
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
# 保存方式3
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x
net = Net()
torch.save(net,"net_method1.pth")