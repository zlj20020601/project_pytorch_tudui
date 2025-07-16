from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("data",train=True,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False,num_workers=0,drop_last=True)
# 它是一个数据加载器，配置用于从 CIFAR10 数据集中加载数据，并且：#
# dataset=dataset: 它将使用前面创建的 CIFAR10 数据集对象作为数据源。#
# batch_size=64: 每次从 dataloader 中迭代时，它会返回一个包含 64 个样本的批次。#
# shuffle=True: 在每个 epoch（训练周期）开始时，数据集中样本的顺序会被打乱。#
# num_workers=0: 数据加载将在主进程中进行，不会启动额外的子进程。#
# drop_last=True: 如果数据集中的样本总数不能被 batch_size (64) 整除，那么最后一个不完整的批次将被丢弃。
# 测试数据集中的第一张图片
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("Dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch),imgs,step)
        # writer.add_images(f"Epoch: {epoch}", imgs, step) 这个是更新的版本
        #这里images 一定要有s
        step = step + 1

writer.close()