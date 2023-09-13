from ultralytics import YOLO
from model.unet import UNet

import lightning as l
from PIL import Image
import numpy as np
import torch

from torch.utils.data import DataLoader
from dataload.dataload import VOCDataset, make_datapath_list, DataTransform
from utils.evaluation import save_mask, mask_iou

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from PIL import Image
from ultralytics import YOLO

# crack detection

def crack_detection(model, crack_tensor):

    results = model.predict("/unet-semantic/images/processing/flight_2_image_results/img/ppm_538_981337489.jpg",
                            save=True, conf = 0.5) # BCHW format with RGB channels float32 (0.0-1.0).
    xywh = results[0].boxes.xywh
    crack_detection_result = []
    for i, data in enumerate(xywh): 
        crack_ = crack_tensor[:, :, ]    
    return crack_detection_result

def crack_segmentation(model, crack_tensor):
    
    model = model
    crack_segmentation_result = model(crack_tensor)
    
    return crack_segmentation_result
    
    
def projection(detection_result, segmenation_result, cfg):   

    boxes = detection_result.boxes
    
    h  = boxes
    w = boxes
    x = boxes
    y = boxes
    H = cfg.input_shape[0]
    W = cfg.input_shape[1]
    
    final_crack_tensor = 0# 关于如何将mask如贴图一样贴上去，建议参考mask和original img贴在一起的程序
    
    
    return final_crack_tensor
    
@hydra.main(version_base=None, config_path='/unet-semantic/config', config_name='cfg')    
def main(cfg: DictConfig):
    
    COLOR_MEAN = (0.485, 0.456, 0.406)
    COLOR_STD = (0.229, 0.224, 0.225)
    
    crack_detection_model = YOLO(cfg.detection_model.pretrained_path)
    crack_segmentation_ = UNet(cfg.segmentation_model.cfg.c_in, cfg.segmentation_model.cfg.c_out)
    crack_segmentation_model = crack_segmentation_.load_from_checkpoint(cfg.segmentation_model.pretrained_path)
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(cfg.data_path)
    test_dataset = VOCDataset(val_img_list, val_anno_list,
                              phase='val',
                               n_classes= 1,
                               input_shape= 475,
                               transform=DataTransform(input_size=256, color_mean=COLOR_MEAN, color_std=COLOR_STD))

    test_dataloader = DataLoader(test_dataset, 
                                  batch_size= 1,
                                  num_workers= 8,
                                #   shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    
    for n_iter, batch in enumerate(test_dataloader):
        
        img, mask = batch
        crack_detection_result = crack_detection(crack_detection_model, img)
        pred_mask = crack_segmentation(crack_segmentation_model, crack_detection_result)
        # final_mask = projection(crack_detection_result, pred_mask, cfg)
        save_mask(pred_mask)
        mask_iou(pred_mask)

if __name__ == "__main__":
    main()
