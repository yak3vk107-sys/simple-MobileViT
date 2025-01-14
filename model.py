import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules import *
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MobileViT(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.batch_first = True
        # 初始特征提取+下采样
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(),
            # nn.Dropout(p=0.1)
        )
        
        # 增加MV2块，无需下采样
        self.mv2 = MV2Block(16, 32, expand_ratio=4, stride=1)
        
        self.mv2_2 = MV2Block(32, 64, expand_ratio=4, stride=2)
        
        # Stage 1: MobileViT块
        self.stage1 = nn.Sequential(
            CustomMobileViTBlock(
                in_channels=64,
                transformer_dim=64,
                ffn_dim=128,
                n_transformer_blocks=2,
                head_dim=16,
                patch_h=2,
                patch_w=2,
                cnn_dropout = 0.1,
                transformer_dropout = 0
            ),
            # nn.Dropout(p=0.1)
        )
        
        # 降采样MV2块
        self.mv2_3 = MV2Block(64, 128, expand_ratio=4, stride=2)
        
        # Stage 2: MobileViT块
        # self.stage2 = nn.Sequential(
        #     CustomMobileViTBlock(
        #         in_channels=32,
        #         transformer_dim=32,
        #         ffn_dim=256,
        #         n_transformer_blocks=3,
        #         head_dim=4,
        #         patch_h=2,
        #         patch_w=2,
        #         cnn_dropout = 0.1,
        #         transformer_dropout = 0
        #     ),
        #     # nn.Dropout(p=0.1)
        # )
        
        # 1x1卷积
        self.conv_last = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
            # nn.Dropout(p=0.1)
        )
        
        # 全局池化和分类器
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 特征提取+下采样
        x = self.conv1(x)        # 32x32 -> 16x16
        
        # MV2块
        x = self.mv2(x)
        x = self.mv2_2(x)
        # MobileViT处理
        x = self.stage1(x)       # 保持分辨率
        
        # MV2下采样
        x = self.mv2_3(x)         # 16x16 -> 8x8
        
        # MobileViT处理
        # x = self.stage2(x)      # 保持分辨率
        
        # 后处理
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x