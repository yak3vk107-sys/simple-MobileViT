import torch
import torch.nn as nn
from typing import Optional, Union

class MV2Block(nn.Module):
    """MobileNetV2 风格的倒置残差块,适用于MobileViT
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数 
        expand_ratio: 扩展比例
        stride: 步长,默认1
        use_norm: 是否使用BN,默认True
        use_act: 是否使用激活函数,默认True
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: Union[int, float],
        stride: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        block = []
        
        # 1. 扩展层
        if expand_ratio != 1:
            block.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim) if use_norm else nn.Identity(),
                nn.ReLU6(inplace=True) if use_act else nn.Identity()
            ])
            
        # 2. 深度可分离卷积
        block.extend([
            # 深度卷积
            nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            bias=False
            ),
            nn.BatchNorm2d(num_features=hidden_dim) if use_norm else nn.Identity(),
            nn.ReLU6(inplace=True) if use_act else nn.Identity()
        ])
        
        # 3. 投影层
        block.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        ])
        
        self.block = nn.Sequential(*block)
        
        # 保存参数用于打印
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_ch={self.in_channels}, out_ch={self.out_channels}, expand={self.expand_ratio}, stride={self.stride})"

    
# 测试代码
if __name__ == "__main__":
    # 测试不同配置的块
    test_configs = [
        # (in_ch, out_ch, expand_ratio, stride, input_size)
        (32, 32, 1, 1, 224),  # 保持维度不变
        (32, 32, 4, 1, 224),  
        (32, 64, 4, 2, 224),  # 降采样
        (64, 128, 4, 2, 112), # 降采样
    ]
    
    for in_ch, out_ch, expand, stride, size in test_configs:
        # 创建块实例
        block = MV2Block(
            in_channels=in_ch,
            out_channels=out_ch,
            expand_ratio=expand,
            stride=stride
        )
        
        # 测试输入 (batch_size, channels, height, width)
        x = torch.randn(1, in_ch, size, size)
        
        # 前向传播
        out = block(x)
        
        print("\nTesting block:")
        print(f"Block config: {block}")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {out.shape}")