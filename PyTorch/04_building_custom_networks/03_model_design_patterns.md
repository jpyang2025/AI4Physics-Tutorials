# 4.3 模型设计模式

## 概述

本节介绍神经网络设计中常用的模式和技巧。这些模式经过多年实践验证，可以帮助你构建更有效的网络。

## 残差连接 (Residual Connection)

### 基本思想

残差连接是深度学习最重要的突破之一。核心思想是让网络学习**残差**（输入与输出的差异），而不是直接学习映射：

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

而不是：

$$\mathbf{y} = F(\mathbf{x})$$

### 物理直觉

残差学习类似于**微扰理论**：
- 恒等映射 $\mathbf{x}$ 是"零阶近似"
- $F(\mathbf{x})$ 是"修正项"
- 网络学习的是相对于恒等映射的偏离

这使得深层网络更容易训练，因为梯度可以通过跳跃连接直接传播。

### 实现

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """基础残差块"""
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # 残差连接
        return self.relu(out)

# 测试
block = ResidualBlock(64)
x = torch.randn(2, 64, 32, 32)
y = block(x)
print(f"输入输出形状相同: {x.shape == y.shape}")
```

### 带下采样的残差块

当输入输出维度不同时，需要投影：

```python
class ResidualBlockWithDownsample(nn.Module):
    """带下采样的残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 当维度不匹配时，需要投影
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = out + residual
        return self.relu(out)

# 下采样示例
block = ResidualBlockWithDownsample(64, 128, stride=2)
x = torch.randn(2, 64, 32, 32)
y = block(x)
print(f"下采样: {x.shape} -> {y.shape}")  # [2, 64, 32, 32] -> [2, 128, 16, 16]
```

## 瓶颈结构 (Bottleneck)

### 基本思想

使用 1×1 卷积先降维，处理后再升维，减少计算量：

```
输入 (256 通道)
    ↓ 1×1 卷积，降到 64
处理 (64 通道)
    ↓ 3×3 卷积
处理 (64 通道)
    ↓ 1×1 卷积，升回 256
输出 (256 通道)
```

### 实现

```python
class Bottleneck(nn.Module):
    """瓶颈残差块"""
    
    expansion = 4  # 输出通道是中间通道的 4 倍
    
    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        
        self.block = nn.Sequential(
            # 1×1 降维
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3×3 卷积
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1×1 升维
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

# 测试
bottleneck = Bottleneck(256, 64)  # 256 -> 64 -> 64 -> 256
x = torch.randn(2, 256, 32, 32)
y = bottleneck(x)
print(f"瓶颈块: {x.shape} -> {y.shape}")
```

## 密集连接 (Dense Connection)

### 基本思想

DenseNet 中的密集连接：每一层都与之前所有层相连。

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])$$

其中 $[\cdot]$ 表示通道维度的拼接。

### 实现

```python
class DenseLayer(nn.Module):
    """密集层"""
    
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)
        )
    
    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)  # 拼接

class DenseBlock(nn.Module):
    """密集块"""
    
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 测试
dense_block = DenseBlock(64, growth_rate=32, num_layers=4)
x = torch.randn(2, 64, 32, 32)
y = dense_block(x)
print(f"密集块: {x.shape} -> {y.shape}")  # 通道从 64 增加到 64 + 4*32 = 192
```

## U-Net 结构

### 基本思想

U-Net 是编码器-解码器结构的经典设计，广泛用于图像分割：

```
编码器路径（下采样）        解码器路径（上采样）
    ↓                           ↑
    ↓ ─────跳跃连接─────→       ↑
    ↓                           ↑
    ↓ ─────跳跃连接─────→       ↑
    ↓                           ↑
        瓶颈层
```

### 实现

```python
class UNet(nn.Module):
    """简化的 U-Net"""
    
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        
        # 编码器
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        
        # 瓶颈
        self.bottleneck = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels)
        
        # 输出
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e3))
        
        # 解码 + 跳跃连接
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.output(d1)

# 测试
unet = UNet(in_channels=1, out_channels=2)
x = torch.randn(2, 1, 256, 256)
y = unet(x)
print(f"U-Net: {x.shape} -> {y.shape}")  # [2, 1, 256, 256] -> [2, 2, 256, 256]
```

**物理应用**：
- 物理场重建
- 图像分割
- 超分辨率

## 注意力机制

### 通道注意力 (SE Block)

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

# 测试
se = SEBlock(64)
x = torch.randn(2, 64, 32, 32)
y = se(x)
print(f"SE 块: {x.shape} -> {y.shape}")
```

### 空间注意力

```python
class SpatialAttention(nn.Module):
    """空间注意力"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算均值和最大值
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        y = torch.cat([avg_pool, max_pool], dim=1)
        # 生成注意力图
        attention = self.sigmoid(self.conv(y))
        return x * attention
```

### CBAM (同时使用通道和空间注意力)

```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```

## 自注意力 (Self-Attention)

### Vision Transformer 风格的自注意力

```python
class SelfAttention2D(nn.Module):
    """图像自注意力"""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 重塑为序列: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        
        # 重塑回图像: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
```

## 多尺度特征融合

### 特征金字塔网络 (FPN)

```python
class SimpleFPN(nn.Module):
    """简化的特征金字塔网络"""
    
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # 1×1 卷积统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        # 3×3 卷积平滑
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        features: 从低分辨率到高分辨率的特征列表
        """
        # 首先应用 lateral 卷积
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = nn.functional.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:], 
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled
        
        # 平滑
        outputs = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        
        return outputs
```

## 条件网络

### 条件批归一化

用于根据条件（如类别标签）调整归一化参数：

```python
class ConditionalBatchNorm2d(nn.Module):
    """条件批归一化"""
    
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        
        # 初始化
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
    
    def forward(self, x, y):
        """
        x: 特征 (B, C, H, W)
        y: 类别标签 (B,)
        """
        out = self.bn(x)
        gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(y).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta
```

## 物理约束网络

### 对称性约束

```python
class SymmetricNetwork(nn.Module):
    """
    输出具有某种对称性的网络
    例如：关于 x=0 的反射对称
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # 使用 |x| 作为输入来保证偶函数性质
        x_symmetric = torch.abs(x)
        return self.network(x_symmetric)

class EquivariantNetwork(nn.Module):
    """
    等变网络示例
    f(-x) = -f(x) (奇函数)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # 奇函数激活
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # 利用 tanh 的奇函数性质
        return self.network(x)
```

### 守恒律约束

```python
class ConservativeNetwork(nn.Module):
    """
    输出满足守恒律的网络
    例如：输出的和为常数
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, target_sum=1.0):
        raw_output = self.network(x)
        # 使用 softmax 确保输出和为 1
        # 或者显式归一化
        normalized = raw_output - raw_output.mean(dim=-1, keepdim=True)
        return normalized + target_sum / raw_output.shape[-1]
```

## 本节小结

| 设计模式 | 核心思想 | 应用场景 |
|---------|---------|---------|
| 残差连接 | 学习残差而非映射 | 深层网络 |
| 瓶颈结构 | 先降维再升维 | 减少计算量 |
| 密集连接 | 所有层互联 | 特征重用 |
| U-Net | 编码-解码+跳跃 | 分割任务 |
| 注意力 | 自适应特征加权 | 增强重要特征 |
| FPN | 多尺度融合 | 目标检测 |
| 条件网络 | 根据条件调整行为 | 条件生成 |

## 练习

1. 实现一个完整的 ResNet-18 网络
2. 将注意力机制添加到 U-Net 中
3. 设计一个保持排列不变性的网络（用于处理点云）

## 延伸阅读

- He et al. (2016). "Deep Residual Learning for Image Recognition"
- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Vaswani et al. (2017). "Attention Is All You Need"

---

[← 上一节：常用层类型](./02_layer_types.md) | [返回章节目录](./README.md)

