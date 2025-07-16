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


image_path = "../imgs/dog.jpg" #绝对路径
