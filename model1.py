import torch
from torch import nn


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
if __name__ == "__main__":
    # 如果脚本是直接运行的：__name__ 这个特殊变量的值会是字符串 __main__。此时，if 语句的条件为真，: 后面的代码块就会被执行。
    # 如果脚本是作为模块被导入的：__name__ 这个特殊变量的值会是该模块的名称（即文件名，不带 .py 后缀）。此时，if 语句的条件为假，: 后面的代码块就不会被执行。
    net = Net()
    input = torch.ones((64,3,32,32))
    output = net(input)
    print(output.shape)
