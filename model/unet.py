
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.spb import SpectralBlock

import lightning as l
from torchmetrics import JaccardIndex


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):

        x = x.reshape(-1, self.channels, self.size * self.size).swapaxes(1, 2).contiguous()
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).contiguous().reshape(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None, 
                 residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )


    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(l.LightningModule):
    def __init__(self, c_in, c_out):
        super(UNet, self).__init__()
    
        self.save_hyperparameters()
        # self.train_loss = nn.BCELoss()
        # self.val_loss = nn.BCELoss()
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        #  self.train_jaccard = JaccardIndex(task='multiclass',
        #                                   threshold=0.5,
        #                                   num_classes=c_out,
        #                                   average='macro')
        
        # self.val_jaccard = JaccardIndex(task='multiclass',
        #                                 threshold=0.5,
        #                                 num_classes=c_out,
        #                                 average='macro')
        # 256*256
        self.c_out = c_out
        self.c_in = c_in
        self.inc = DoubleConv(self.c_in, 64) # 3, 256, 256 -> 64, 256, 256
        self.down1 = Down(64, 128)           # 64, 256, 256 -> 128, 128, 128
        self.sa1 = SelfAttention(128, 128)   # 128, 128, 128 -> 128, 128, 128
        self.down2 = Down(128, 256)          # 128, 128, 128 -> 256, 64, 64
        self.sa2 = SelfAttention(256, 64)    # 256, 64, 64 -> 256, 64, 64
        self.down3 = Down(256, 256)          # 256, 64, 64 -> 256, 32, 32
        self.sa3 = SelfAttention(256, 32)    # 256, 32, 32 -> 256, 32, 32

        self.bot1 = DoubleConv(256, 512)     # 256, 32, 32 -> 512, 32, 32
        self.bot2 = DoubleConv(512, 512)     # 512, 32, 32 -> 512, 32, 32
        self.bot3 = DoubleConv(512, 256)     # 512, 32, 32 -> 256, 32, 32

        self.up1 = Up(512, 128)              # 256, 32, 32 -> 128, 64, 64 
        self.sa4 = SelfAttention(128, 64)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 128)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 256)
        self.outc = nn.Conv2d(64, self.c_out, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.2)

        self.spb_128 = SpectralBlock(128, 128, 128)
        self.spb_256 = SpectralBlock(256, 64, 64)
        self.spb_512 = SpectralBlock(512, 32, 32)        
        
    def training_step(self, batch, batch_idx):

        imgs, pngs = batch
        # print(pngs.shape)
        # pngs = pngs.unsqueeze(1)
        pngs = pngs.long()

        x1 = self.inc(imgs)
        x2 = self.down1(x1)
        x2 = self.spb_128(x2)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.spb_256(x3)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3)
        # x4 = self.sa3(x4)


        x4 = self.bot1(x4)
        x4 = self.spb_512(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        # x = self.sa4(x)
        x = self.up2(x, x2)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        # x = self.sa6(x)
        x = self.dropout(x)
        output = self.outc(x)
        # output = self.sigmoid(output)

        # print(output.shape)
        # print(pngs.shape)
        # output = torch.squeeze(output)
        # print(output.shape)
        self.train_loss_value = self.train_loss(output, pngs)
        # self.train_miou = self.train_jaccard(output, pngs)
        
        
        # self.log_experiment.
        # self.logger.experiment.log_confusion_matrix(
        #     y_true = pngs[None,...].detach(),
        #     y_predicted = output.detach(),
        #     images = imgs
        # )
        # values = {'train_loss': train_loss,
        #           'train_miou': train_miou}
        # self.log_dict(values,
        #               sync_dist=True,
        #               on_step=True,
        #               on_epoch=True,
        #               logger=True)
        # print(self.train_loss_value)
        return self.train_loss_value
    def on_train_epoch_end(self):
        
        self.logger.log_metrics({'train_loss': self.train_loss_value,
                                #  'train_miou': self.train_miou
                                 })
        
    
    def validation_step(self, batch, batch_idx):
        
        imgs, pngs = batch
        pngs = pngs.long()
        # pngs = pngs.unsqueeze(1)
        
        x1 = self.inc(imgs)

        x2 = self.down1(x1)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3)
        # x4 = self.sa3(x4)


        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        # x = self.sa4(x)
        x = self.up2(x, x2)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        # x = self.sa6(x)
        
        output = self.outc(x)
        # output = self.sigmoid(output)
        
        self.val_loss_value = self.val_loss(output, pngs)
        # self.val_miou = self.val_jaccard(output, pngs)

        # values = {'val_loss': val_loss,
        #           'val_miou': val_miou}
        
        # self.log_dict(values,
        #               sync_dist=True,
        #               on_step=True,
        #               on_epoch=True,
        #               logger=True)
        
        
        return self.val_loss_value
    def on_validation_epoch_end(self):
        
        self.logger.log_metrics({'val_loss': self.val_loss_value,
                                #  'val_miou': self.val_miou
                                 })        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9,0.999),eps=1e-8)
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return optimizer
    
    def forward(self, batch):
        imgs= batch

        x1 = self.inc(imgs)
        x2 = self.down1(x1)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3)
        # x4 = self.sa3(x4)


        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        # x = self.sa4(x)
        x = self.up2(x, x2)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        # x = self.sa6(x)
        output = self.outc(x)

        return output