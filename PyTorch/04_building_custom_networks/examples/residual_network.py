#!/usr/bin/env python3
"""
残差网络 (ResNet) 实现

本脚本演示如何从零实现 ResNet 及其变体，
并解释残差连接的物理直觉。

运行方式：
    python residual_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type
import math


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 基础残差块
# ============================================================

class BasicBlock(nn.Module):
    """
    基础残差块（用于 ResNet-18/34）
    
    结构:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    
    物理类比：微扰论
    y = x + F(x)
    网络学习的是相对于恒等映射的"修正"
    """
    
    expansion = 1  # 输出通道相对于中间通道的倍数
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        # 第一个卷积
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接（当维度不匹配时需要投影）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差连接的核心
        out = self.relu(out)
        
        return out


# ============================================================
# 2. 瓶颈残差块
# ============================================================

class Bottleneck(nn.Module):
    """
    瓶颈残差块（用于 ResNet-50/101/152）
    
    结构:
    x -> 1x1 Conv (降维) -> 3x3 Conv -> 1x1 Conv (升维) -> (+x) -> ReLU
    
    优点：减少计算量，同时保持表达能力
    """
    
    expansion = 4  # 输出通道是中间通道的 4 倍
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        out_channels = mid_channels * self.expansion
        
        # 1x1 降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 3x3 卷积
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 1x1 升维
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================
# 3. 完整的 ResNet
# ============================================================

class ResNet(nn.Module):
    """
    完整的 ResNet 实现
    
    支持:
    - ResNet-18: layers=[2,2,2,2], BasicBlock
    - ResNet-34: layers=[3,4,6,3], BasicBlock
    - ResNet-50: layers=[3,4,6,3], Bottleneck
    - ResNet-101: layers=[3,4,23,3], Bottleneck
    - ResNet-152: layers=[3,8,36,3], Bottleneck
    """
    
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        zero_init_residual: bool = True
    ):
        super().__init__()
        
        self.in_channels = 64
        
        # Stem (输入层)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: Type[nn.Module],
        channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """构建残差层"""
        
        downsample = None
        out_channels = channels * block.expansion
        
        # 当步长不为1或通道数改变时，需要下采样
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # 第一个块可能改变维度
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = out_channels
        
        # 剩余块维度不变
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # 将残差分支的最后一个 BN 初始化为零
        # 这样初始时网络就是恒等映射
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不经过分类器）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# 工厂函数
def resnet18(num_classes: int = 1000, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)

def resnet34(num_classes: int = 1000, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet50(num_classes: int = 1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)

def resnet101(num_classes: int = 1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def demo_resnet():
    """演示 ResNet"""
    print_section("3. 完整 ResNet 实现")
    
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ResNet-34': resnet34(num_classes=10),
        'ResNet-50': resnet50(num_classes=10),
    }
    
    print("模型参数统计:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 参数")
    
    # 测试 ResNet-18
    model = models['ResNet-18']
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    features = model.get_features(x)
    
    print(f"\nResNet-18 前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  特征: {features.shape}")


# ============================================================
# 4. 小型 ResNet（适用于小图像如 CIFAR）
# ============================================================

class SmallResNet(nn.Module):
    """
    小型 ResNet，适用于 32x32 图像（如 CIFAR-10）
    
    区别于标准 ResNet:
    - 第一层使用 3x3 卷积而非 7x7
    - 去掉 maxpool
    - 使用更少的通道数
    """
    
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
        base_channels: int = 16
    ):
        super().__init__()
        
        self.in_channels = base_channels
        
        # 简化的 stem
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, 3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, channels, num_blocks, stride=1):
        downsample = None
        out_channels = channels * block.expansion
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def small_resnet20(num_classes: int = 10) -> SmallResNet:
    """CIFAR-10 的 ResNet-20"""
    return SmallResNet(BasicBlock, [3, 3, 3], num_classes, base_channels=16)

def small_resnet56(num_classes: int = 10) -> SmallResNet:
    """CIFAR-10 的 ResNet-56"""
    return SmallResNet(BasicBlock, [9, 9, 9], num_classes, base_channels=16)


def demo_small_resnet():
    """演示小型 ResNet"""
    print_section("4. 小型 ResNet（CIFAR）")
    
    model = small_resnet20(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-20 参数量: {params:,}")
    
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"前向传播: {x.shape} -> {y.shape}")


# ============================================================
# 5. 残差连接的梯度分析
# ============================================================

def demo_gradient_flow():
    """演示残差连接对梯度流的影响"""
    print_section("5. 残差连接的梯度分析")
    
    # 创建两个网络：有/无残差连接
    class PlainBlock(nn.Module):
        """无残差连接的块"""
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x
    
    class ResBlock(nn.Module):
        """有残差连接的块"""
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
        
        def forward(self, x):
            identity = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x = F.relu(x + identity)  # 残差连接
            return x
    
    # 构建深层网络
    num_blocks = 20
    channels = 32
    
    plain_net = nn.Sequential(*[PlainBlock(channels) for _ in range(num_blocks)])
    res_net = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
    
    # 测试梯度
    x = torch.randn(1, channels, 8, 8, requires_grad=True)
    
    # Plain 网络
    x_plain = x.clone()
    y_plain = plain_net(x_plain)
    y_plain.sum().backward()
    grad_plain = x_plain.grad.norm().item()
    
    # Res 网络
    x_res = x.clone().detach().requires_grad_(True)
    y_res = res_net(x_res)
    y_res.sum().backward()
    grad_res = x_res.grad.norm().item()
    
    print(f"网络深度: {num_blocks} 个块")
    print(f"Plain 网络输入梯度范数: {grad_plain:.6f}")
    print(f"ResNet 输入梯度范数: {grad_res:.6f}")
    print(f"梯度比值 (Res/Plain): {grad_res/grad_plain:.2f}x")
    print("\n残差连接显著改善了深层网络的梯度流！")


# ============================================================
# 6. 预激活 ResNet
# ============================================================

class PreActBlock(nn.Module):
    """
    预激活残差块
    
    结构: BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+x)
    
    优点：更好的梯度流，更容易训练更深的网络
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(x))
        
        identity = out if self.downsample is None else self.downsample(out)
        
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        
        return out + identity


def demo_preact_resnet():
    """演示预激活 ResNet"""
    print_section("6. 预激活 ResNet")
    
    block = PreActBlock(64, 64)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    
    print("预激活块结构:")
    print("  BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+x)")
    print(f"\n前向传播: {x.shape} -> {y.shape}")
    print("\n优点: 更好的梯度流，适合训练非常深的网络")


# ============================================================
# 7. 物理应用：等变残差网络
# ============================================================

class EquivariantResBlock(nn.Module):
    """
    保持某种对称性的残差块
    
    示例：平移等变（通过卷积保证）
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 使用卷积保证平移等变性
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


def demo_physics_resnet():
    """演示物理相关的 ResNet 应用"""
    print_section("7. 物理应用：等变残差网络")
    
    print("残差网络在物理中的应用:")
    print("  1. 平移等变性：通过卷积层自动保证")
    print("  2. 能量守恒：残差连接类似于能量的增量更新")
    print("  3. 微扰论视角：y = x + F(x)，学习"修正项"")
    
    # 验证平移等变性
    block = EquivariantResBlock(32)
    x = torch.randn(1, 32, 16, 16)
    x_shifted = torch.roll(x, shifts=2, dims=3)
    
    y = block(x)
    y_shifted_direct = block(x_shifted)
    y_shifted_from_y = torch.roll(y, shifts=2, dims=3)
    
    error = (y_shifted_direct - y_shifted_from_y).abs().mean().item()
    print(f"\n平移等变性验证:")
    print(f"  f(shift(x)) vs shift(f(x)) 差异: {error:.6f}")
    print("  （接近零表示具有平移等变性）")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" 残差网络 (ResNet) 实现")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 演示基础块
    print_section("1. 基础残差块")
    block = BasicBlock(64, 64)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    print(f"BasicBlock: {x.shape} -> {y.shape}")
    
    print_section("2. 瓶颈残差块")
    bottleneck = Bottleneck(256, 64)
    x = torch.randn(4, 256, 32, 32)
    y = bottleneck(x)
    print(f"Bottleneck: {x.shape} -> {y.shape}")
    
    # 运行其他演示
    demo_resnet()
    demo_small_resnet()
    demo_gradient_flow()
    demo_preact_resnet()
    demo_physics_resnet()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 残差连接: y = x + F(x)，学习残差而非映射")
    print("  2. 瓶颈结构: 1x1降维 -> 3x3卷积 -> 1x1升维")
    print("  3. 残差连接显著改善深层网络的梯度流")
    print("  4. 预激活ResNet: BN-ReLU-Conv 顺序更优")
    print("  5. 物理视角: 类似微扰论，学习相对于恒等的修正")
    print()


if __name__ == "__main__":
    main()

