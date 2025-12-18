# 3.1 torchvision 模型库

## torchvision 简介

`torchvision` 是 PyTorch 官方的计算机视觉库，提供：

1. **预训练模型**：在 ImageNet 等大型数据集上训练好的模型
2. **数据集**：常用的视觉数据集（MNIST、CIFAR、ImageNet 等）
3. **图像变换**：数据预处理和增强工具
4. **实用工具**：可视化、IO 等

```python
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

print(f"torchvision 版本: {torchvision.__version__}")
```

## 常用预训练模型

### 模型架构概览

#### 1. ResNet (Residual Network)

**核心思想**：残差连接（Skip Connection）

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

这解决了深层网络的梯度消失问题，使得训练非常深的网络成为可能。

```python
# 加载 ResNet
resnet18 = models.resnet18(weights='IMAGENET1K_V1')  # 11M 参数
resnet50 = models.resnet50(weights='IMAGENET1K_V1')  # 25M 参数
resnet101 = models.resnet101(weights='IMAGENET1K_V1')  # 44M 参数

print(f"ResNet-18 参数量: {sum(p.numel() for p in resnet18.parameters()):,}")
```

**物理类比**：残差连接类似于物理中的**微扰论**——网络学习的是相对于恒等映射的"修正"。

#### 2. VGG

**核心思想**：简单堆叠 3×3 卷积层

```python
vgg16 = models.vgg16(weights='IMAGENET1K_V1')  # 138M 参数
vgg19 = models.vgg19(weights='IMAGENET1K_V1')  # 143M 参数

print(f"VGG-16 参数量: {sum(p.numel() for p in vgg16.parameters()):,}")
```

VGG 结构简单直观，常用于特征提取和教学。

#### 3. EfficientNet

**核心思想**：复合缩放（同时优化深度、宽度、分辨率）

```python
efficientnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1')  # 5M 参数
efficientnet_b7 = models.efficientnet_b7(weights='IMAGENET1K_V1')  # 66M 参数

print(f"EfficientNet-B0 参数量: {sum(p.numel() for p in efficientnet_b0.parameters()):,}")
```

EfficientNet 在参数效率和精度之间取得了很好的平衡。

#### 4. Vision Transformer (ViT)

**核心思想**：将 Transformer 架构应用于图像

```python
vit_b_16 = models.vit_b_16(weights='IMAGENET1K_V1')  # 86M 参数

print(f"ViT-B/16 参数量: {sum(p.numel() for p in vit_b_16.parameters()):,}")
```

ViT 将图像分成 patch，像处理序列一样处理图像。

#### 5. ConvNeXt

**核心思想**：用现代技术（如 LayerNorm、GELU）改进 CNN

```python
convnext_tiny = models.convnext_tiny(weights='IMAGENET1K_V1')  # 29M 参数

print(f"ConvNeXt-Tiny 参数量: {sum(p.numel() for p in convnext_tiny.parameters()):,}")
```

### 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 快速原型 | ResNet-18 | 小巧快速 |
| 平衡精度与速度 | ResNet-50, EfficientNet-B0 | 性价比高 |
| 最高精度 | EfficientNet-B7, ViT-L | 精度最好 |
| 特征提取 | VGG-16, ResNet | 特征直观 |
| 边缘部署 | MobileNet, EfficientNet-B0 | 参数少 |
| 大数据集 | ViT, ConvNeXt | 缩放性好 |

## 加载预训练模型

### 新 API（PyTorch 2.0+）

```python
import torchvision.models as models

# 使用 weights 参数指定预训练权重
# IMAGENET1K_V1: ImageNet-1K 训练的权重（第一版）
# IMAGENET1K_V2: 更新的训练方法，精度更高
# DEFAULT: 最新的默认权重
# None: 不加载预训练权重（随机初始化）

resnet50 = models.resnet50(weights='IMAGENET1K_V1')
resnet50_v2 = models.resnet50(weights='IMAGENET1K_V2')
resnet50_default = models.resnet50(weights='DEFAULT')
resnet50_random = models.resnet50(weights=None)

# 查看可用的权重
from torchvision.models import ResNet50_Weights
print(ResNet50_Weights.IMAGENET1K_V1)
print(ResNet50_Weights.IMAGENET1K_V2)
```

### 旧 API（仍然支持）

```python
# 使用 pretrained 参数（已弃用但仍可用）
resnet50 = models.resnet50(pretrained=True)  # 等价于 weights='IMAGENET1K_V1'
resnet50_random = models.resnet50(pretrained=False)  # 等价于 weights=None
```

## 查看模型结构

### 打印模型

```python
resnet18 = models.resnet18(weights='IMAGENET1K_V1')

# 打印完整结构
print(resnet18)

# 只看顶层模块
for name, module in resnet18.named_children():
    print(f"{name}: {module.__class__.__name__}")
```

输出示例：
```
conv1: Conv2d
bn1: BatchNorm2d
relu: ReLU
maxpool: MaxPool2d
layer1: Sequential
layer2: Sequential
layer3: Sequential
layer4: Sequential
avgpool: AdaptiveAvgPool2d
fc: Linear
```

### 查看参数

```python
# 查看所有参数
for name, param in resnet18.named_parameters():
    print(f"{name:50} {str(list(param.shape)):20} {param.numel():>10}")

# 统计参数量
total_params = sum(p.numel() for p in resnet18.parameters())
trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
print(f"\n总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
```

### 使用 torchinfo 可视化

```python
# pip install torchinfo
from torchinfo import summary

resnet18 = models.resnet18(weights='IMAGENET1K_V1')
summary(resnet18, input_size=(1, 3, 224, 224))
```

## 模型的输入输出

### 输入要求

大多数 torchvision 模型期望的输入格式：

- **形状**：`(batch_size, 3, H, W)`
- **通道**：RGB 顺序
- **尺寸**：通常 224×224（某些模型支持其他尺寸）
- **归一化**：ImageNet 均值和标准差

```python
# 标准的图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),           # 短边缩放到256
    transforms.CenterCrop(224),       # 中心裁剪224×224
    transforms.ToTensor(),            # 转为张量 [0, 1]
    transforms.Normalize(             # ImageNet 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 使用 weights 自带的预处理（推荐）
from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
```

### 输出格式

分类模型输出 **logits**（未归一化的分数），形状为 `(batch_size, num_classes)`：

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# 假设输入
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    logits = model(x)

print(f"输出形状: {logits.shape}")  # [1, 1000]

# 转换为概率
probs = torch.softmax(logits, dim=1)
print(f"概率和: {probs.sum().item():.4f}")  # 1.0

# 获取预测类别
pred_class = logits.argmax(dim=1)
print(f"预测类别索引: {pred_class.item()}")
```

## ImageNet 类别

ImageNet-1K 有 1000 个类别。可以从权重对象获取类别名称：

```python
from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]

print(f"类别数量: {len(categories)}")
print(f"前10个类别: {categories[:10]}")

# 示例：根据索引获取类别名
class_idx = 281
print(f"类别 {class_idx}: {categories[class_idx]}")  # tabby cat
```

或者从文件加载：

```python
# 从 URL 下载类别列表
import urllib.request
import json

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# urllib.request.urlretrieve(url, "imagenet_classes.txt")

# 也可以硬编码常见类别
SAMPLE_CATEGORIES = {
    0: "tench",
    1: "goldfish",
    281: "tabby cat",
    282: "tiger cat",
    285: "Egyptian cat",
    # ...
}
```

## 卷积神经网络简介

如果你不熟悉 CNN，这里简要介绍核心概念：

### 卷积层 (Convolutional Layer)

卷积层用**滤波器**（kernel）在图像上滑动，提取局部特征：

$$y[i,j] = \sum_{m,n} x[i+m, j+n] \cdot k[m,n]$$

```python
import torch.nn as nn

# 卷积层示例
conv = nn.Conv2d(
    in_channels=3,     # 输入通道（RGB=3）
    out_channels=64,   # 输出通道（特征图数量）
    kernel_size=3,     # 滤波器大小 3×3
    stride=1,          # 步长
    padding=1          # 填充
)

x = torch.randn(1, 3, 224, 224)
y = conv(x)
print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
# 输入: [1, 3, 224, 224], 输出: [1, 64, 224, 224]
```

**物理类比**：卷积类似于信号处理中的滤波器，或物理中的格林函数卷积。

### 池化层 (Pooling Layer)

池化层降低空间分辨率，增加感受野：

```python
# 最大池化
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 64, 224, 224)
y = maxpool(x)
print(f"池化后形状: {y.shape}")  # [1, 64, 112, 112]
```

### 批归一化 (Batch Normalization)

标准化每个特征通道，加速训练：

```python
bn = nn.BatchNorm2d(64)
y = bn(x)
```

**物理类比**：类似于将变量标准化到相同尺度。

### 典型 CNN 结构

```
输入图像 (3, 224, 224)
    ↓
[卷积 → BN → ReLU → 池化] × N  -- 特征提取
    ↓
展平 (Flatten)
    ↓
全连接层 (FC)  -- 分类
    ↓
输出 (1000,)
```

## 模型保存与加载

### 保存整个模型

```python
# 保存
torch.save(model, 'model.pth')

# 加载
model = torch.load('model.pth')
```

### 只保存权重（推荐）

```python
# 保存权重
torch.save(model.state_dict(), 'model_weights.pth')

# 加载权重
model = models.resnet18()  # 先创建模型结构
model.load_state_dict(torch.load('model_weights.pth'))
```

### 从 Hugging Face Hub 加载

```python
# 许多模型也可以从 Hugging Face Hub 加载
# pip install timm
import timm

# 列出可用模型
# print(timm.list_models('resnet*'))

# 加载模型
model = timm.create_model('resnet50', pretrained=True)
```

## 本节小结

| 概念 | 说明 |
|------|------|
| torchvision.models | PyTorch 官方预训练模型库 |
| weights 参数 | 指定预训练权重版本 |
| ImageNet | 1000 类图像分类数据集 |
| 输入格式 | (B, 3, 224, 224)，需归一化 |
| 输出格式 | (B, 1000) logits |

## 练习

1. 加载 EfficientNet-B0，统计其参数量，并与 ResNet-50 对比
2. 查看 VGG-16 的结构，理解其"简单堆叠"的设计
3. 使用 torchinfo 可视化一个模型，观察每层的输入输出形状

## 延伸阅读

- [torchvision.models 官方文档](https://pytorch.org/vision/stable/models.html)
- He et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

[返回章节目录](./README.md) | [下一节：模型推理 →](./02_model_inference.md)

