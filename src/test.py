# import torch
# outputs = torch.tensor([
#     [0.1,0.2],
#     [0.05,0.4],
# ])
# print(outputs.argmax(1)) #argmax()中的0以列判断  1代表以行判断
# preds = outputs.argmax(1)
# targets = torch.tensor([0,1])
# print(preds == targets)
# print((preds == targets).sum())
import torch
from PIL import Image
from torch import nn
from torchvision import  transforms

image_path = "../imgs/dogs.png" #绝对路径
image = Image.open(image_path)
# png格式是四个通道，除了RGB三通道外，还有一个透明度通道
image = image.convert("RGB")
image.show()
transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

image = transform(image)
# print(image.size())
print(image.shape)
# 属性访问 (Attribute Access)
# 属性（Attribute）**是对象的数据。它们是存储在对象内部的变量，描述了对象的状态或特征。#
# 如何访问： 你通过点运算符 .后面直接跟属性的名称来访问它。
# print(image.shape) image是一个Tensor对象，shape是它的一个属性
# 方法访问 (Method Access / Method Call)
# 方法（Method）是对象的行为或功能。它们是定义在类中的函数，可以对对象的属性进行操作，或者执行与对象相关的某些动作。
# 如何访问（调用）： 你通过点运算符 . 后面跟方法的名称，然后紧接着一对括号 () 来调用它。如果方法需要参数，参数就放在括号里。
# print(image.size()) # image是一个Tensor对象，size()是它的一个方法
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("model_9.pth ",map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))