import lightning as l
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
from model.unet import UNet
from dataload.dataload import VOCDataset, make_datapath_list, DataTransform

matplotlib.use('TkAgg')

rootpath = "/unet-semantic/data/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=rootpath)

# Generate Dataset
# (RGB)颜色的平均值和标准差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", n_classes=21, input_shape=[256, 256],
                         transform=DataTransform(
    input_size=256, color_mean=color_mean, color_std=color_std, ))

model = UNet(c_in=3,c_out=21)
model = model.load_from_checkpoint("/unet-semantic/check_point/epoch=499-val_loss=0.00-train_miou=0.00.ckpt",
                                   map_location='cpu')
model.cuda()
model.eval()
print('网络设置完毕 ：成功载入了训练完毕的权重。')


img_index = 5

# 1.显示原有图像
image_file_path = val_img_list[img_index]
img_original = Image.open(image_file_path)  # [高度][宽度][颜色RGB]
img_width, img_height = img_original.size
plt.imshow(img_original)
plt.savefig("img_original.png")
plt.show()

# 2.创建预处理类
anno_file_path = val_anno_list[img_index]
anno_class_img = Image.open(anno_file_path)   # [高度][宽度][颜色RGB]
# p_palette = anno_class_img.getpalette() # maybe something wrong with this
plt.imshow(anno_class_img)
plt.savefig("anno_class_img.png")
plt.show()

# 3. 用UNET进行推论
img, anno_class_img = val_dataset.__getitem__(img_index)
img = img.cuda()
anno_class_img =  anno_class_img.cuda()
x = img.unsqueeze(0)  # 小批量化：torch.Size([1, 3, 256, 256])
outputs = model(x)
y = outputs  # 忽略AuxLoss

# 4. 从uNet的输出结果求取最大分类，并转换为颜色调色板格式，将图像尺寸恢复为原有尺寸
y = y[0].detach().cpu().numpy()  # y：torch.Size([1, 21, 475, 475])
y = np.argmax(y, axis=0)
anno_class_img = Image.fromarray(np.uint8(y), mode="P")
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
plt.imshow(anno_class_img)
plt.savefig("anno_class_img.png")
plt.show()

# 5.将图像透明化并重叠在一起
trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
anno_class_img = anno_class_img.convert('RGBA')

for x in range(img_width):
    for y in range(img_height):
        # 获取推测结果的图像的像素数据
        pixel = anno_class_img.getpixel((x, y))
        r, g, b, a = pixel

        # 如果是(0, 0, 0)的背景，直接透明化
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue
        else:
            # 将除此之外的颜色写入准备好的图像中
            trans_img.putpixel((x, y), (r, g, b, 200))
            # 150指定的是透明度大小

result = Image.alpha_composite(img_original.convert('RGBA'), trans_img)
plt.imshow(result)
plt.savefig("result.png")
plt.show()
