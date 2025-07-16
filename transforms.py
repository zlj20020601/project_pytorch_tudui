from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

img_path = "archive/ants/707895295_009cf23188.jpg"
writer = SummaryWriter(log_dir="logs")
cv2_img = cv2.imread(img_path)
# print(cv2_img.shape)
img = Image.open(img_path)
print(img)
# img.show
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# tensor_img1 = transforms.ToTensor(img)


print(tensor_img)
writer.add_image("tensor_image", tensor_img, 0)
writer.close()