import torch
import torch.nn.functional as F

from model import output

input = torch.tensor([
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,3,1,1],
    [2,1,0,1,1]
])
kernel = torch.tensor([
    [1,2,1],
    [0,1,0],
    [2,1,0]
])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
# 使用张量自带的 .reshape() 方法并重新赋值
# input = input.reshape((1,1,5,5))
# kernel = kernel.reshape((1,1,3,3))

print(input.shape)
print(kernel.shape)
output = F.conv2d(input, kernel,stride=1,bias=None)
print(output)

output2 = F.conv2d(input, kernel,stride=2,bias=None)
print(output2)

output3 = F.conv2d(input, kernel,stride=2,bias=None,padding=1)
print(output3)
# torch.nn.Conv2d (作为类)
# 位置： torch.nn 模块下。
#
# 用途： 用于在定义神经网络模型时，作为模型的一个层 (Layer)。当你创建 nn.Module 的子类时，会实例化 nn.Conv2d。
#
# 参数 bias： 在 nn.Conv2d 的构造函数中，bias 参数确实是 bool 类型。如果你设置为 True (默认值)，
# nn.Conv2d 内部会自动创建并管理一个可训练的偏置张量 (learnable bias tensor)。你不需要手动创建这个偏置张量。
# 这个层会作为模型的一部分，它的参数（包括权重和偏置）会自动被优化器发现和更新。
# import torch.nn as nn
# conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=True)
# # conv_layer 内部会自动创建一个 bias 参数 (torch.nn.Parameter)

# torch.nn.functional.conv2d (作为函数，通常导入为 F.conv2d)
#
# 位置： torch.nn.functional 模块下。
#
# 用途： 提供低级、无状态的函数式操作。它不创建层，不管理参数。你调用它时，必须手动提供所有必需的输入张量和权重张量。
#
# 参数 bias： 在 F.conv2d 函数中，bias 参数期望的是一个张量 (Tensor)。如果你想应用偏置，你需要自己创建一个偏置张量并传入。
# 如果你不想应用偏置，你可以传入 None 或者干脆不写这个参数（因为 F.conv2d 的 bias 参数默认值是 None）。
# import torch.nn.functional as F
# import torch
#
# input_tensor = torch.randn(1, 1, 5, 5)
# weight_tensor = torch.randn(1, 1, 3, 3)
# bias_tensor = torch.randn(1) # 手动创建一个偏置张量
#
# output = F.conv2d(input_tensor, weight_tensor, bias=bias_tensor) # 传入偏置张量
# # 或者不带偏置
# output_no_bias = F.conv2d(input_tensor, weight_tensor) # bias 默认为 None