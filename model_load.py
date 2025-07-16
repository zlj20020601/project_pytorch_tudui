import torch
import torchvision
# import model_save
from model_save import Net
# method1
# model = torch.load("vgg16_method1.pth")
# print(model)

# method 2
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
# print(model2)

model3 = torch.load("net_method1.pth")#必须import model_save
print(model3)

