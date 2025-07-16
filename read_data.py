from torch.utils.data import Dataset  # 导入dataset
import cv2
#导入 OpenCV 库，OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉库，提供了许多功能来处理图像和视频。
# 通过这个库，你可以执行图像读取、处理、视频分析、对象检测等任务
import os
from PIL import Image
import  numpy as np

# img_path = "D:\\Pycharm\\project\\archive\\ants\\707895295_009cf23188.jpg"
# img = Image.open(img_path)
# img.show()
# dir_path = "D:\\Pycharm\\project\\archive\\ants" #定义了一个名为 dir_path 的字符串变量。
# img_path_list = os.listdir(dir_path)  #os.listdir(dir_path) 函数的作用是列出指定目录 dir_path 下的所有文件和子目录的名称，它会返回一个字符串列表，其中每个字符串都是目录中的一个文件或子目录的名称。这个列表被赋值给变量 img_path_list
# img1 = img_path_list[0]
# print(img1)

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = r"D:\Pycharm\project\archive"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_label_dir = "bees"
bees_dataset= MyData(root_dir,bees_label_dir)
train_dataset = ants_dataset + bees_dataset
img, label = train_dataset[26]
img.show()

# import os
# from PIL import Image
# from torch.utils.data import Dataset
#
# class MyDataWithSeparateLabels(Dataset):
#
#     def __init__(self, root_dir, class_name, img_ext=".jpg", label_ext=".txt"):
#         self.root_dir = root_dir
#         self.class_name = class_name
#         self.img_ext = img_ext # 图片文件后缀，例如 ".jpg", ".png"
#         self.label_ext = label_ext # 标签文件后缀，例如 ".txt", ".json"
#
#         # 实际图片所在的文件夹路径
#         self.image_folder_path = os.path.join(self.root_dir, self.class_name + "_image")
#         # 实际标签所在的文件夹路径
#         self.label_folder_path = os.path.join(self.root_dir, self.class_name + "_label")
#
#         # 获取所有图片文件名（不含路径）
#         self.img_names = [f for f in os.listdir(self.image_folder_path) if f.endswith(self.img_ext)]
#         # 可选：如果需要确保图片和标签一一对应，可以在这里进行检查
#         # self.label_names = [f for f in os.listdir(self.label_folder_path) if f.endswith(self.label_ext)]
#         # assert len(self.img_names) == len(self.label_names), "图片和标签文件数量不匹配！"
#
#     def __getitem__(self, index):
#         img_name = self.img_names[index]
#
#         # 完整图片路径
#         img_item_path = os.path.join(self.image_folder_path, img_name)
#         img = Image.open(img_item_path).convert("RGB")
#
#         # 根据图片名推断标签文件名（假设同名，后缀不同）
#         base_name = os.path.splitext(img_name)[0] # 获取不带后缀的文件名
#         label_file_name = base_name + self.label_ext
#         label_file_path = os.path.join(self.label_folder_path, label_file_name)
#
#         # 加载标签数据
#         # 这一部分需要根据你标签文件的具体格式来修改
#         label_data = None
#         try:
#             with open(label_file_path, 'r', encoding='utf-8') as f:
#                 label_data = f.read().strip() # 读取文件内容作为标签
#                 # 如果标签是数字，可能需要转换为int或float
#                 # label_data = int(label_data)
#         except FileNotFoundError:
#             print(f"警告: 未找到 {label_file_path} 对应的标签文件，将使用默认标签或跳过。")
#             # 可以选择抛出错误，或者返回一个默认标签，或者跳过这个样本
#             label_data = -1 # 例如，用-1表示缺失标签
#
#         return img, label_data
#
#     def __len__(self):
#         return len(self.img_names)
#
# # --- 使用示例 ---
# root_dir = "D:\\Pycharm\\project\\archive"
#
# # 假设蚂蚁图片是.jpg，标签是.txt
# ants_dataset_with_labels = MyDataWithSeparateLabels(root_dir=root_dir, class_name="ants", img_ext=".jpg", label_ext=".txt")
# print(f"蚂蚁数据集图片数量: {len(ants_dataset_with_labels)}")
# img_ant, label_data_ant = ants_dataset_with_labels[0]
# print(f"第一张蚂蚁图片路径: {os.path.join(ants_dataset_with_labels.image_folder_path, ants_dataset_with_labels.img_names[0])}")
# print(f"第一张蚂蚁图片加载的标签数据: {label_data_ant}")
#
# # 假设蜜蜂图片是.png，标签是.json
# # bees_dataset_with_labels = MyDataWithSeparateLabels(root_dir=root_dir, class_name="bees", img_ext=".png", label_ext=".json")
# # ... 类似地使用