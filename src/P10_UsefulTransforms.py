from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img = Image.open("../archive/bees/16838648_415acd9e3f.jpg")
print(img)
# ToTensor
trans_ToTensor = transforms.ToTensor()
img_tensor = trans_ToTensor(img)
writer.add_image("img_tensor", img_tensor, 0)
print(img_tensor[0][0][0])

# Normalize

trans_Normalize = transforms.Normalize([1, 3, 5],[0.5, 0.5, 0.5])
img_norm= trans_Normalize(img_tensor)
print(img_norm[0][0][0])
writer.add_image("img_norm_tensor", img_norm, 1)

# Resize

print(img.size)
trans_Resize = transforms.Resize((256, 512))
img_Resize = trans_Resize(img)
img_Resize = trans_ToTensor(img_Resize)
print(img_Resize)
writer.add_image("img_Resize", img_Resize, 1)

# Compose
trans_Resize_2 = transforms.Resize(512)
# PIL ->PIL  ->Tensor
trans_compose = transforms.Compose([trans_Resize_2 , trans_ToTensor])
img_compose = trans_compose(img)
print(img_compose)
writer.add_image("img_compose", img_compose, 0)

# RandomCrop
trans_random = transforms.RandomCrop(256)
trans_Compose_2 = transforms.Compose([trans_random , trans_ToTensor])
for i in range(10):
    img_crop = trans_Compose_2(img)
    writer.add_image("img_crop", img_crop, i)
writer.close()