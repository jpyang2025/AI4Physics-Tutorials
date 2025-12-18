# 3.3 迁移学习

## 什么是迁移学习？

**迁移学习**（Transfer Learning）是将在一个任务上学到的知识应用到另一个相关任务的技术。核心思想是：

> 在大规模数据上预训练的模型学到了通用的特征表示，这些特征可以迁移到新任务。

### 为什么迁移学习有效？

1. **特征的层次性**：
   - 底层特征（边缘、纹理）是通用的
   - 高层特征逐渐变得任务特定

2. **数据效率**：
   - 新任务可能只有少量数据
   - 预训练提供了良好的初始化

3. **训练效率**：
   - 不需要从头训练整个网络
   - 只需微调部分层

### 物理类比

迁移学习类似于物理中的**微扰理论**：

- **预训练模型** = 已知系统的精确解（如氢原子）
- **新任务** = 稍有不同的系统（如类氢原子）
- **微调** = 微扰修正
- **冻结层** = 保持主导项不变

## 迁移学习的两种策略

### 策略1：特征提取 (Feature Extraction)

使用预训练模型作为固定的特征提取器，只训练新的分类头。

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
num_features = model.fc.in_features
num_classes = 10  # 新任务的类别数
model.fc = nn.Linear(num_features, num_classes)

# 此时只有 model.fc 的参数需要训练
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
```

**优点**：
- 训练快速
- 需要的数据少
- 不容易过拟合

**缺点**：
- 特征可能不完全适合新任务
- 精度可能不如完全微调

### 策略2：微调 (Fine-tuning)

在预训练权重的基础上，对部分或全部层进行训练。

```python
# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# 替换分类头
num_features = model.fc.in_features
num_classes = 10
model.fc = nn.Linear(num_features, num_classes)

# 方法1：微调所有层（使用较小的学习率）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 方法2：分层学习率
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# 方法3：只微调最后几层
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**优点**：
- 通常能达到更高精度
- 特征适应新任务

**缺点**：
- 需要更多数据和计算
- 可能过拟合
- 需要仔细调参

## 实践：图像分类迁移学习

### 准备数据

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 训练数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证/测试预处理
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 假设数据目录结构：
# data/
#   train/
#     class1/
#     class2/
#   val/
#     class1/
#     class2/

# train_dataset = ImageFolder('data/train', transform=train_transform)
# val_dataset = ImageFolder('data/val', transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

### 构建模型

```python
import torch.nn as nn
import torchvision.models as models

def create_transfer_model(num_classes, freeze_backbone=True, model_name='resnet50'):
    """
    创建迁移学习模型
    
    参数:
        num_classes: 新任务的类别数
        freeze_backbone: 是否冻结主干网络
        model_name: 使用的预训练模型
    """
    # 选择模型
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        num_features = model.classifier[6].in_features
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 冻结主干网络
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 替换分类头
    if model_name.startswith('resnet'):
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    elif model_name.startswith('efficientnet'):
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    elif model_name.startswith('vgg'):
        model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

# 创建模型
model = create_transfer_model(num_classes=10, freeze_backbone=True)
print(f"模型结构的最后几层:\n{list(model.children())[-1]}")
```

### 训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """完整训练流程"""
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 只优化需要梯度的参数
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    return model
```

## 渐进式解冻

一种高级微调策略：逐步解冻更多的层。

```python
def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    解冻模型的最后 N 层
    """
    # 获取所有层的名称
    layer_names = [name for name, _ in model.named_parameters()]
    
    # 解冻最后 N 层
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= len(layer_names) - num_layers_to_unfreeze:
            param.requires_grad = True
            print(f"解冻: {name}")
        else:
            param.requires_grad = False

def progressive_unfreezing_training(model, train_loader, val_loader, device):
    """渐进式解冻训练"""
    
    # 阶段1：只训练分类头
    print("阶段1：只训练分类头")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    # 训练几个 epoch...
    
    # 阶段2：解冻最后一个残差块
    print("\n阶段2：解冻 layer4")
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    # 训练几个 epoch...
    
    # 阶段3：解冻更多层
    print("\n阶段3：解冻 layer3 和 layer4")
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    # 训练几个 epoch...
```

## 特征提取

将预训练模型用作特征提取器，获取图像的特征向量：

```python
import torch
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """特征提取器"""
    
    def __init__(self, model_name='resnet50', layer_name='avgpool'):
        super().__init__()
        
        # 加载预训练模型
        if model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif model_name == 'vgg16':
            self.model = models.vgg16(weights='IMAGENET1K_V1')
        
        self.model.eval()
        
        # 存储特征的钩子
        self.features = None
        self.layer_name = layer_name
        
        # 注册钩子
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._hook)
    
    def _hook(self, module, input, output):
        self.features = output
    
    def forward(self, x):
        with torch.no_grad():
            _ = self.model(x)
        return self.features.flatten(1)  # 展平为向量

# 使用特征提取器
extractor = FeatureExtractor('resnet50', 'avgpool')

x = torch.randn(8, 3, 224, 224)
features = extractor(x)
print(f"特征形状: {features.shape}")  # [8, 2048]
```

### 使用特征进行下游任务

```python
# 提取特征后，可以使用简单的分类器
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 假设已经提取了特征
# train_features: [N, 2048]
# train_labels: [N]

# 使用 SVM 分类
# svm = SVC(kernel='rbf')
# svm.fit(train_features.numpy(), train_labels.numpy())

# 使用逻辑回归
# lr = LogisticRegression(max_iter=1000)
# lr.fit(train_features.numpy(), train_labels.numpy())
```

## 物理应用：显微镜图像分类

这是一个物理科研中常见的应用场景：

```python
"""
示例：使用迁移学习分类显微镜图像

假设任务：区分不同晶体结构的显微镜图像
- 类别：立方晶系、六方晶系、四方晶系等
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class CrystalClassifier(nn.Module):
    """晶体结构分类器"""
    
    def __init__(self, num_crystal_types=5):
        super().__init__()
        
        # 使用预训练的 EfficientNet
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # 冻结主干网络
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 自定义分类头（针对灰度图像可能需要调整）
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_crystal_types)
        )
    
    def forward(self, x):
        return self.backbone(x)

# 显微镜图像预处理（可能需要特殊处理）
microscope_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # 如果是灰度图像，需要复制到3通道
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # 使用数据集自己的统计量，或者使用 ImageNet 的
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建分类器
classifier = CrystalClassifier(num_crystal_types=5)
print(f"晶体分类器可训练参数: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
```

## 迁移学习最佳实践

### 1. 数据量与策略选择

| 数据量 | 与原任务相似度 | 推荐策略 |
|--------|--------------|---------|
| 少 | 高 | 特征提取（冻结所有层） |
| 少 | 低 | 特征提取 + 简单分类器 |
| 多 | 高 | 微调最后几层 |
| 多 | 低 | 从头训练或微调更多层 |

### 2. 学习率选择

```python
# 经验法则
# - 预训练层：1e-5 ~ 1e-4
# - 新添加的层：1e-3 ~ 1e-2

optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### 3. 数据增强

对于小数据集，数据增强尤为重要：

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])
```

### 4. 防止过拟合

```python
# 使用 Dropout
classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, num_classes)
)

# 使用权重衰减（L2 正则化）
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 早停
# 如果验证损失连续 N 个 epoch 不下降，停止训练
```

## 本节小结

| 概念 | 说明 |
|------|------|
| 迁移学习 | 将预训练知识迁移到新任务 |
| 特征提取 | 冻结主干，只训练分类头 |
| 微调 | 解冻部分/全部层进行训练 |
| 渐进式解冻 | 逐步解冻更多的层 |
| 分层学习率 | 不同层使用不同学习率 |

## 练习

1. 使用预训练的 ResNet-18 对 CIFAR-10 数据集进行迁移学习
2. 比较特征提取和微调两种策略的效果
3. 实现渐进式解冻训练，观察每个阶段的性能变化

## 延伸阅读

- Yosinski et al. (2014). "How transferable are features in deep neural networks?"
- [PyTorch 迁移学习教程](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Howard & Ruder (2018). "Universal Language Model Fine-tuning for Text Classification"

---

[← 上一节：模型推理](./02_model_inference.md) | [返回章节目录](./README.md)

