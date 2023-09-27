import lightning as l
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
from model.unet import UNet
from dataload.dataload import VOCDataset, make_datapath_list, DataTransform
from torch.utils.data import DataLoader
# matplotlib.use('TkAgg')

rootpath = "/unet-semantic/data/crack_seg/"
train_img_list, train_anno_list, val_img_list, val_anno_list, _, _ = make_datapath_list(
    rootpath=rootpath)

# Generate Dataset
# (RGB)颜色的平均值和标准差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

test_dataset = VOCDataset(val_img_list, val_anno_list, phase="test", n_classes=21, input_shape=[256, 256],
                         transform=DataTransform(
    input_size=256, color_mean=color_mean, color_std=color_std, ))
test_dataloader = DataLoader(test_dataset, 
                                  batch_size=1, # train_config['batch_size'],
                                  num_workers=8, # train_config['num_workers'],
                                #   shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
model = UNet(c_in=3,c_out=2)
model = model.load_from_checkpoint("/unet-semantic/check_point/last-v2.ckpt",
                                   map_location='cpu')

# model.cuda()
model.eval()
print('网络设置完毕 ：成功载入了训练完毕的权重。')

for n, batch in enumerate(test_dataloader):
    img_index = n

    # 1.显示原有图像

    image_file_path = val_img_list[img_index]
    img_original = Image.open(image_file_path)  # [高度][宽度][颜色RGB]
    img_width, img_height = img_original.size
    print(img_width)
    print(img_height)
    plt.imshow(img_original)
    # plt.savefig("img_original.png")
    img_original.save(f'{n}'+  'img_original.png')
    plt.show()

    # 2.创建预处理类
    anno_file_path = val_anno_list[img_index]
    anno_class_img = Image.open(anno_file_path)   # [高度][宽度][颜色RGB]
    # p_palette = anno_class_img.getpalette() # maybe something wrong with this

    plt.imshow(anno_class_img)
    # plt.savefig("anno_class_img_origin.png")
    anno_class_img.save(f'{n}'+'anno_class_img_origin.png')
    plt.show()

    # 3. 用UNET进行推论
    img, anno_class_img = batch 
    outputs = model(img)
    y = outputs  # 忽略AuxLoss

    # 4. 从uNet的输出结果求取最大分类，并转换为颜色调色板格式，将图像尺寸恢复为原有尺寸
    # y = y[0].detach().cpu().numpy()  # y：torch.Size([1, 21, 475, 475])
    y = y[0].detach().cpu().numpy()
    a = np.unique(y)
    print(a)
    y = np.argmax(y, axis=0)
    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    print(anno_class_img.size)
    plt.imshow(anno_class_img)
    # plt.savefig("anno_class_img.png")
    anno_class_img.save(f'{n}'+'pred.png')
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
    # plt.savefig("result.png")
    result.save(f'{n}'+'img_preds.png')
    plt.show()
