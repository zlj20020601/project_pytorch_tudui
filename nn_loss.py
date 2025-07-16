import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))# batch_size channel height width
targets = torch.reshape(targets,(1,1,1,3))

loss = L1Loss(reduction='sum')
result = loss(inputs,targets)
loss_mse = nn.MSELoss(reduction='sum')
result1 = loss_mse(inputs,targets)
print(result)
print(result1)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])#在 CrossEntropyLoss 的上下文中，y 代表了真实标签 (true label)。
# 这里的 1 表示这个样本的真实类别是索引为 1 的那个类别。记住，在 PyTorch 中，类别索引通常是从 0 开始的。所以，1 指的是第二个类别。
x = torch.reshape(x,(1,3))#这个形状 (batch_size, num_classes) 是 nn.CrossEntropyLoss 期望的 input 张量的典型形状。
loss_cross = nn.CrossEntropyLoss ()
result = loss_cross(x,y)
print(result)