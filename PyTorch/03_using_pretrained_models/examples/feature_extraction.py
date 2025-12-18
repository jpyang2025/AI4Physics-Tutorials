#!/usr/bin/env python3
"""
特征提取与迁移学习示例

本脚本演示如何使用预训练模型进行特征提取和迁移学习，
包括冻结层、微调和渐进式解冻。

运行方式：
    python feature_extraction.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Dict, List, Tuple


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 特征提取器
# ============================================================

class FeatureExtractor(nn.Module):
    """
    通用特征提取器
    
    可以从预训练模型的任意层提取特征
    """
    
    def __init__(self, model_name: str = 'resnet50', layer_name: str = 'avgpool'):
        super().__init__()
        
        # 加载预训练模型
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
        elif model_name == 'vgg16':
            self.model = models.vgg16(weights='IMAGENET1K_V1')
            self.feature_dim = 4096
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model.eval()
        self.layer_name = layer_name
        self.features = None
        
        # 注册前向钩子
        self._register_hook()
    
    def _register_hook(self):
        """注册钩子以捕获中间层输出"""
        def hook(module, input, output):
            self.features = output
        
        # 找到目标层并注册钩子
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                print(f"钩子注册到: {name} ({module.__class__.__name__})")
                return
        
        raise ValueError(f"找不到层: {self.layer_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        with torch.no_grad():
            _ = self.model(x)
        
        # 展平特征
        if self.features.dim() > 2:
            return self.features.flatten(1)
        return self.features
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.feature_dim


def demo_feature_extraction():
    """演示特征提取"""
    print_section("1. 特征提取")
    
    # 创建特征提取器
    extractor = FeatureExtractor('resnet50', 'avgpool')
    
    # 创建随机输入
    x = torch.randn(4, 3, 224, 224)
    
    # 提取特征
    features = extractor(x)
    
    print(f"输入形状: {x.shape}")
    print(f"特征形状: {features.shape}")
    print(f"特征维度: {extractor.get_feature_dim()}")
    print(f"特征统计: 均值={features.mean():.4f}, 标准差={features.std():.4f}")
    
    return extractor


# ============================================================
# 2. 迁移学习模型
# ============================================================

class TransferLearningModel(nn.Module):
    """
    迁移学习模型
    
    支持特征提取（冻结主干）和微调两种模式
    """
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50', 
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.freeze_backbone = freeze_backbone
        
        # 加载预训练主干网络
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # 移除原分类头
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的主干网络: {backbone}")
        
        # 冻结主干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 新的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.in_features = in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.classifier(features)
    
    def unfreeze_backbone(self, num_layers: int = None):
        """解冻主干网络的部分或全部层"""
        params = list(self.backbone.parameters())
        
        if num_layers is None:
            # 解冻所有层
            for param in params:
                param.requires_grad = True
            print(f"解冻所有 {len(params)} 个参数组")
        else:
            # 只解冻最后 N 个参数
            for param in params[-num_layers:]:
                param.requires_grad = True
            print(f"解冻最后 {num_layers} 个参数组")
    
    def get_trainable_params(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """返回总参数数量"""
        return sum(p.numel() for p in self.parameters())


def demo_transfer_model():
    """演示迁移学习模型"""
    print_section("2. 迁移学习模型")
    
    # 创建模型（冻结主干）
    model_frozen = TransferLearningModel(num_classes=10, backbone='resnet50', freeze_backbone=True)
    
    print("冻结主干网络时:")
    print(f"  总参数: {model_frozen.get_total_params():,}")
    print(f"  可训练参数: {model_frozen.get_trainable_params():,}")
    print(f"  可训练比例: {model_frozen.get_trainable_params()/model_frozen.get_total_params()*100:.2f}%")
    
    # 解冻部分层
    model_frozen.unfreeze_backbone(num_layers=20)
    print(f"\n解冻最后20个参数组后:")
    print(f"  可训练参数: {model_frozen.get_trainable_params():,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    y = model_frozen(x)
    print(f"\n前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model_frozen


# ============================================================
# 3. 模拟数据集和训练
# ============================================================

def create_synthetic_dataset(num_samples: int = 500, num_classes: int = 5):
    """创建合成数据集用于演示"""
    print_section("3. 创建合成数据集")
    
    # 创建随机"图像"数据（实际应用中应该是真实图像）
    torch.manual_seed(42)
    
    # 简单的合成数据：每个类别有不同的颜色偏差
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # 为每个类别添加特征
    for c in range(num_classes):
        mask = y == c
        X[mask, c % 3] += 2.0  # 某个通道加偏置
    
    # 划分训练/验证集
    n_train = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练样本: {n_train}")
    print(f"验证样本: {num_samples - n_train}")
    print(f"类别数: {num_classes}")
    
    return train_loader, val_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
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
    
    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total


def demo_training():
    """演示迁移学习训练"""
    print_section("4. 迁移学习训练演示")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_loader, val_loader = create_synthetic_dataset(num_samples=200, num_classes=5)
    
    # 创建模型
    model = TransferLearningModel(num_classes=5, backbone='resnet18', freeze_backbone=True)
    model = model.to(device)
    
    print(f"\n模型可训练参数: {model.get_trainable_params():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    # 训练
    print("\n开始训练（特征提取模式）:")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
    
    return model


# ============================================================
# 5. 渐进式解冻训练
# ============================================================

def demo_progressive_unfreezing():
    """演示渐进式解冻"""
    print_section("5. 渐进式解冻训练")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集
    train_loader, val_loader = create_synthetic_dataset(num_samples=200, num_classes=5)
    
    # 创建模型（完全冻结）
    model = TransferLearningModel(num_classes=5, backbone='resnet18', freeze_backbone=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 阶段1：只训练分类头
    print("阶段1：只训练分类头")
    print(f"可训练参数: {model.get_trainable_params():,}")
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    for epoch in range(3):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2%}")
    
    # 阶段2：解冻部分主干网络
    print("\n阶段2：解冻最后10个参数组")
    model.unfreeze_backbone(num_layers=10)
    print(f"可训练参数: {model.get_trainable_params():,}")
    
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    for epoch in range(3):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2%}")
    
    # 阶段3：解冻全部
    print("\n阶段3：解冻全部网络")
    model.unfreeze_backbone()
    print(f"可训练参数: {model.get_trainable_params():,}")
    
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    for epoch in range(3):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2%}")
    
    return model


# ============================================================
# 6. 特征可视化
# ============================================================

def demo_feature_visualization():
    """演示特征统计"""
    print_section("6. 特征分析")
    
    # 创建特征提取器
    extractor = FeatureExtractor('resnet18', 'avgpool')
    
    # 创建不同"类别"的输入
    torch.manual_seed(42)
    
    # 模拟两类数据
    class_a = torch.randn(10, 3, 224, 224)
    class_a[:, 0] += 3.0  # 红色偏置
    
    class_b = torch.randn(10, 3, 224, 224)
    class_b[:, 2] += 3.0  # 蓝色偏置
    
    # 提取特征
    features_a = extractor(class_a)
    features_b = extractor(class_b)
    
    print("特征统计:")
    print(f"  类别 A: 均值={features_a.mean():.4f}, 标准差={features_a.std():.4f}")
    print(f"  类别 B: 均值={features_b.mean():.4f}, 标准差={features_b.std():.4f}")
    
    # 计算类间距离
    center_a = features_a.mean(dim=0)
    center_b = features_b.mean(dim=0)
    class_distance = torch.norm(center_a - center_b)
    print(f"  类中心距离: {class_distance:.4f}")
    
    # 类内距离
    intra_a = torch.norm(features_a - center_a, dim=1).mean()
    intra_b = torch.norm(features_b - center_b, dim=1).mean()
    print(f"  类 A 内距离: {intra_a:.4f}")
    print(f"  类 B 内距离: {intra_b:.4f}")
    
    # Fisher 判别比
    fisher = class_distance / (intra_a + intra_b)
    print(f"  Fisher 判别比: {fisher:.4f}")


# ============================================================
# 7. 不同层的特征对比
# ============================================================

def demo_layer_comparison():
    """对比不同层的特征"""
    print_section("7. 不同层特征对比")
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    
    # 需要提取特征的层
    layers_to_check = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    features_dict = {}
    
    # 注册钩子
    def get_hook(name):
        def hook(module, input, output):
            features_dict[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if name in layers_to_check:
            module.register_forward_hook(get_hook(name))
    
    # 前向传播
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        _ = model(x)
    
    # 显示每层的特征形状
    print("各层特征形状:")
    for name, feat in features_dict.items():
        print(f"  {name:15} {str(list(feat.shape)):25} 元素数: {feat.numel()//4:,}")


# ============================================================
# 8. 物理应用示例：光谱分类
# ============================================================

def demo_physics_application():
    """物理应用示例：模拟光谱分类"""
    print_section("8. 物理应用：光谱分类模拟")
    
    print("场景：使用迁移学习分类不同材料的光谱图像")
    print()
    
    # 模拟光谱数据（转换为图像格式）
    torch.manual_seed(42)
    
    num_samples = 100
    num_materials = 4
    
    # 模拟不同材料的"光谱特征"
    # 在实际应用中，这些可能是光谱图或其他科学图像
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, num_materials, (num_samples,))
    
    # 为每种材料添加独特特征
    material_signatures = [
        [1.0, 0.0, 0.0],   # 材料1：红色特征
        [0.0, 1.0, 0.0],   # 材料2：绿色特征
        [0.0, 0.0, 1.0],   # 材料3：蓝色特征
        [1.0, 1.0, 0.0],   # 材料4：黄色特征
    ]
    
    for i in range(num_materials):
        mask = y == i
        for c in range(3):
            X[mask, c] += material_signatures[i][c] * 2.0
    
    print(f"数据集: {num_samples} 个样本, {num_materials} 种材料")
    
    # 创建迁移学习模型
    model = TransferLearningModel(
        num_classes=num_materials,
        backbone='resnet18',
        freeze_backbone=True
    )
    
    # 简单训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    print("\n训练光谱分类器:")
    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
    
    print(f"\n最终准确率: {train_acc:.2%}")
    print("\n结论：预训练的图像特征可以迁移到科学图像分类任务")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" PyTorch 特征提取与迁移学习示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # 运行所有演示
    demo_feature_extraction()
    demo_transfer_model()
    demo_training()
    demo_progressive_unfreezing()
    demo_feature_visualization()
    demo_layer_comparison()
    demo_physics_application()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 特征提取：冻结主干，只训练分类头（快速，需要数据少）")
    print("  2. 微调：解冻部分/全部层（精度更高，需要更多数据）")
    print("  3. 渐进式解冻：逐步解冻更多层（平衡策略）")
    print("  4. 分层学习率：底层用小学习率，顶层用大学习率")
    print("  5. 预训练特征可以迁移到科学图像任务")
    print()


if __name__ == "__main__":
    main()

