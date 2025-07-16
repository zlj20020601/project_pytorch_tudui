import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="data",train=True,download=True,transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="data",train=False,download=True,transform=dataset_transform)

# print(len(train_set))
# print(len(test_set))
# print(test_set[0])
# print(test_set.classes)
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
print(test_set[0])
writer = SummaryWriter("P10")
for i in range(10):
    img,target = train_set[i]
    writer.add_image("test_set",img,i)
writer.close()