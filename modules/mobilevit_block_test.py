import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)  # 修改为 [1, max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class CustomMobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,              
        transformer_dim: int,          
        ffn_dim: int,                 
        n_transformer_blocks: int = 2, 
        head_dim: int = 32,           
        transformer_dropout: float = 0.0, 
        cnn_dropout: float = 0.0,     
        patch_h: int = 8,             
        patch_w: int = 8,             
        conv_kernel_size: int = 3,    
        **kwargs                      
    ):
        super().__init__()
        
        # 保存所有参数
        self.in_channels = in_channels
        self.transformer_dim = transformer_dim
        self.ffn_dim = ffn_dim
        self.n_transformer_blocks = n_transformer_blocks
        self.head_dim = head_dim
        self.transformer_dropout = transformer_dropout
        self.cnn_dropout = cnn_dropout
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = transformer_dim // head_dim
        
        # 更新网络层
        self._build_layers()
    
    def _build_layers(self):
        # 1. 局部特征提取
        self.local_rep = nn.Sequential(
            nn.Conv2d(
                self.in_channels, 
                self.in_channels,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cnn_dropout),
            
            nn.Conv2d(
                self.in_channels,
                self.transformer_dim, 
                kernel_size=1,
                bias=False
            )
        )
        
        # 修改位置编码的初始化位置
        self.pos_encoder = PositionalEncoding(
            d_model=self.transformer_dim,
            max_len=64  # 根据实际 patch 数量设置
        )
        
        # 3. 全局特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.transformer_dropout,
            batch_first=True  # 设置batch_first=True
        )
        self.global_rep = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_transformer_blocks
        )
        
        # 4. 特征重投影
        self.proj = nn.Sequential(
            nn.Conv2d(
                self.transformer_dim,
                self.in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cnn_dropout)
        )
        
        # 5. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.in_channels * 2,
                self.in_channels,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cnn_dropout)
        )
            
        # 设置patch_area属性
        self.patch_area = self.patch_h * self.patch_w
            
    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            feature_map = F.interpolate(
                feature_map, 
                size=(new_h, new_w), 
                mode="bilinear", 
                align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P]
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, f"Tensor should be of shape BPxNxC. Got: {patches.shape}"
        
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], 
            self.patch_area, 
            info_dict["total_patches"], 
            -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, 
            num_patch_w, 
            self.patch_h, 
            self.patch_w
        )
        
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, 
            channels, 
            num_patch_h * self.patch_h, 
            num_patch_w * self.patch_w
        )

        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map
    
    def update_parameters(self, **kwargs):
        """更新模型参数并重建网络层"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_layers()

    def forward(self, x):
        # 保存残差
        res = x
        
        # 局部特征提取
        x = self.local_rep(x)
        
        # 转换为patches并记录形状
        patches, info_dict = self.unfolding(x)
        B, N, C = patches.shape
        
        # 重新排列维度以适应位置编码
        patches = patches.reshape(B, N, C)
        patches = self.pos_encoder(patches)
        patches = self.global_rep(patches)
        
        # 重构特征图
        x = self.folding(patches, info_dict)
        
        # 投影
        x = self.proj(x)
        
        # 融合
        x = self.fusion(torch.cat([res, x], dim=1))
        
        return x

# Example usage
# 创建 MobileViT 块实例
# mvit_block1 = CustomMobileViTBlock(
#     in_channels=64,
#     transformer_dim=64,
#     ffn_dim=256,
#     n_transformer_blocks=3,
#     head_dim=32,
#     transformer_dropout=0.1,
#     cnn_dropout=0.1,
#     patch_h=2,
#     patch_w=2,
#     conv_kernel_size=3
# )

# # 创建输入张量
# input_tensor = torch.randn(1, 64, 224, 224)  # Batch size of 1, 64 input channels, 224x224 image

# # 前向传播
# output = mvit_block1(input_tensor)

# # 打印输出形状
# print(output.shape)  # 应该打印 torch.Size([1, 64, 224, 224])
