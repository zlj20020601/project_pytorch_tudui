import torchvision
from torchvision import transforms
from torch import nn

# train_data = torchvision.datasets.ImageNet(root="../data_imageNet", train=True, download=True, transform=transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)
print(vgg16_True)
print('ok')
# 迁移学习
vgg16_True.add_module('add_Linear',nn.Linear(1000,10))
print(vgg16_True)
vgg16_True.classifier.add_module('add_Linear',nn.Linear(1000,10))
# vgg16_false.classifier[6]= nn.Linear(4096,10)