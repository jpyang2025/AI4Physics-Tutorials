#!/usr/bin/env python3
"""
自定义卷积神经网络 (CNN) 示例

本脚本演示如何构建各种 CNN 架构，
包括经典 CNN、现代 CNN 和用于物理数据的专用 CNN。

运行方式：
    python custom_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 基础卷积块
# ============================================================

class ConvBlock(nn.Module):
    """
    基础卷积块: Conv -> BatchNorm -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def demo_conv_block():
    """演示卷积块"""
    print_section("1. 基础卷积块")
    
    block = ConvBlock(in_channels=3, out_channels=64)
    
    x = torch.randn(4, 3, 32, 32)
    y = block(x)
    
    print(f"卷积块结构:\n{block}")
    print(f"\n输入: {x.shape} -> 输出: {y.shape}")
    
    # 计算输出尺寸
    print(f"\n使用 stride=2 下采样:")
    block_downsample = ConvBlock(64, 128, stride=2)
    y2 = block_downsample(y)
    print(f"  {y.shape} -> {y2.shape}")


# ============================================================
# 2. 简单 CNN 分类器
# ============================================================

class SimpleCNN(nn.Module):
    """
    简单的 CNN 图像分类器
    
    结构：
    Conv blocks (特征提取) -> Global Avg Pool -> FC (分类)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32
    ):
        super().__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4),
            nn.MaxPool2d(2),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 4, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不经过分类器）"""
        x = self.features(x)
        x = self.global_pool(x)
        return x.flatten(1)


def demo_simple_cnn():
    """演示简单 CNN"""
    print_section("2. 简单 CNN 分类器")
    
    model = SimpleCNN(in_channels=3, num_classes=10, base_channels=32)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    features = model.get_features(x)
    
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  特征: {features.shape}")


# ============================================================
# 3. VGG 风格网络
# ============================================================

class VGGBlock(nn.Module):
    """VGG 风格的块：多个 3x3 卷积 + 池化"""
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ConvBlock(in_ch, out_channels))
        layers.append(nn.MaxPool2d(2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGGStyleNet(nn.Module):
    """VGG 风格的网络"""
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        config: List[Tuple[int, int]] = None  # [(channels, num_convs), ...]
    ):
        super().__init__()
        
        if config is None:
            config = [(64, 2), (128, 2), (256, 3), (512, 3)]
        
        # 构建特征提取器
        blocks = []
        prev_channels = in_channels
        for channels, num_convs in config:
            blocks.append(VGGBlock(prev_channels, channels, num_convs))
            prev_channels = channels
        
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def demo_vgg_style():
    """演示 VGG 风格网络"""
    print_section("3. VGG 风格网络")
    
    # 小型 VGG
    model = VGGStyleNet(
        in_channels=3,
        num_classes=10,
        config=[(32, 2), (64, 2), (128, 3)]
    )
    
    print(f"网络结构:\n{model.features}")
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print(f"\n前向传播: {x.shape} -> {y.shape}")


# ============================================================
# 4. 现代 CNN（带注意力）
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        attention = self.excitation(self.squeeze(x)).view(b, c, 1, 1)
        return x * attention


class ModernConvBlock(nn.Module):
    """现代卷积块：带 SE 注意力的残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # 残差连接
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        return self.relu(out + residual)


class ModernCNN(nn.Module):
    """现代 CNN：使用 SE 注意力和残差连接"""
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64
    ):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 主体
        self.stage1 = self._make_stage(base_channels, base_channels, 2, stride=1)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, 2, stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, 2, stride=2)
        
        # 头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_classes)
        )
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        blocks = [ModernConvBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(ModernConvBlock(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def demo_modern_cnn():
    """演示现代 CNN"""
    print_section("4. 现代 CNN（带 SE 注意力）")
    
    model = ModernCNN(in_channels=3, num_classes=10, base_channels=32)
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"前向传播: {x.shape} -> {y.shape}")
    
    # 展示 SE 注意力效果
    print("\nSE 注意力权重示例:")
    se = SEBlock(64)
    test_input = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        att = se.excitation(se.squeeze(test_input))
    print(f"  前5个通道权重: {att[0, :5].tolist()}")


# ============================================================
# 5. 用于 1D 信号的 CNN
# ============================================================

class Conv1DNet(nn.Module):
    """
    1D CNN 用于处理一维信号
    
    物理应用：
    - 时间序列
    - 光谱数据
    - 波形分析
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        signal_length: int = 1000
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length) 或 (batch, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        x = self.features(x)
        x = self.classifier(x)
        return x


def demo_1d_cnn():
    """演示 1D CNN"""
    print_section("5. 1D CNN（信号处理）")
    
    model = Conv1DNet(in_channels=1, num_classes=5, signal_length=1000)
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟光谱数据
    x = torch.randn(8, 1000)  # 8个光谱，每个1000个波长点
    y = model(x)
    print(f"前向传播: {x.shape} -> {y.shape}")
    
    # 带通道的输入
    x_multichannel = torch.randn(8, 3, 1000)  # 3通道
    model_multichannel = Conv1DNet(in_channels=3, num_classes=5)
    y_multi = model_multichannel(x_multichannel)
    print(f"多通道: {x_multichannel.shape} -> {y_multi.shape}")


# ============================================================
# 6. U-Net 风格网络（图像分割）
# ============================================================

class DoubleConv(nn.Module):
    """双重卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    """简化的 U-Net"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        # 编码器
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # 输出
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e3))
        
        # 解码 + 跳跃连接
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


def demo_unet():
    """演示 U-Net"""
    print_section("6. U-Net（图像分割）")
    
    model = SimpleUNet(in_channels=1, out_channels=2, base_channels=32)
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 1, 128, 128)
    y = model(x)
    print(f"前向传播: {x.shape} -> {y.shape}")
    print("(输入输出尺寸相同，用于像素级预测)")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" 自定义 CNN 示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 运行所有演示
    demo_conv_block()
    demo_simple_cnn()
    demo_vgg_style()
    demo_modern_cnn()
    demo_1d_cnn()
    demo_unet()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 基础卷积块: Conv -> BN -> ReLU")
    print("  2. VGG 风格: 堆叠 3x3 卷积")
    print("  3. 现代 CNN: 残差连接 + 注意力")
    print("  4. 1D CNN: 适用于信号/序列数据")
    print("  5. U-Net: 编码-解码 + 跳跃连接")
    print()


if __name__ == "__main__":
    main()

