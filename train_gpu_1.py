import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from model1 import Net
import torch.nn as nn
# from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# #preprocessing
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.ToTensor())#数据下载过后会在当前目录的data文件夹下
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
# print(f"训练集的数据长度为：{train_data_size}") f string 的方法
print(f"测试集的长度为：{test_data_size}")
# train=true表示导入train这个训练集
# 利用Dataloader加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建网络模型
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

net = Net()
if torch.cuda.is_available():
    net = net.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = optim.SGD(net.parameters(),lr=learning_rate)


# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epochs = 10
writer = SummaryWriter(log_dir='./log_model1')
start_time = time.time()
for i in range(epochs):
    print("------第{}轮训练开始------".format(i+1))
    # 训练步骤开始
    net.train()
    for data in train_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，损失为{}".format(total_train_step, loss.item()))
            writer.add_scalar("loss", loss.item(), total_train_step)
    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    for data in test_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        total_test_loss += loss.item()
        accuracy = (outputs.argmax(1) == labels).sum()
        total_accuracy += accuracy
        total_test_step += 1
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("loss", total_test_loss, total_test_step)
    writer.add_scalar("accuracy", total_accuracy/test_data_size, total_test_step)

writer.close()



# test_data_iter = iter(testloader)
# test_image, test_label = next(test_data_iter)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))
# net = LeNet()
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
#
# for epoch in range(5):
#     running_loss = 0.0
#     for step,data in enumerate(trainloader,start=0):
#         inputs, labels = data
#         optimizer.zero_grad() #清除历史梯度
#         outputs = net(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()  #损失反向传播
#         optimizer.step()  #更新step这个参数
#         running_loss += loss.item() #item只要值
#         if step % 500 == 499: #print every 500 mini-batches
#             with torch.no_grad():
#                 outputs = net(test_image) #[batch,20]
#                 predict_y = torch.max(outputs, 1)[1]
#                 accuracy = (predict_y == test_label).to(torch.int).sum().item() / test_label.size(0)
#                 print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
#                       (epoch+1, step+1, running_loss/500, accuracy))
#                 running_loss = 0.0
#
#
# print('Finished Training')
# save_path ='./lenet.pth'
# torch.save(net.state_dict(), save_path)


