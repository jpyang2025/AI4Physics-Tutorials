#!/usr/bin/env python3
"""
MNIST 分类完整训练流程

本示例展示了完整的深度学习训练流程：
1. 数据加载与预处理
2. 模型定义
3. 训练循环
4. 验证与测试
5. 模型保存与加载

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


# =============================================================================
# 1. 配置与设备设置
# =============================================================================

class Config:
    """训练配置"""
    # 数据
    data_dir = './data'
    batch_size = 64
    num_workers = 4
    
    # 模型
    input_size = 28 * 28
    hidden_sizes = [512, 256, 128]
    num_classes = 10
    dropout = 0.2
    
    # 训练
    learning_rate = 0.001
    num_epochs = 20
    weight_decay = 1e-4
    
    # 其他
    seed = 42
    save_dir = './checkpoints'


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =============================================================================
# 2. 数据加载
# =============================================================================

def get_data_loaders(config):
    """创建数据加载器"""
    
    # 数据变换
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 统计量
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载数据集
    full_train_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=transform_train,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,
        train=False,
        transform=transform_test,
        download=True
    )
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"数据集大小:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# 3. 模型定义
# =============================================================================

class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # 展平图像
        x = x.view(x.size(0), -1)
        return self.net(x)


class CNN(nn.Module):
    """卷积神经网络"""
    
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 4. 训练器类
# =============================================================================

class Trainer:
    """训练器"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        self.best_val_acc = 0.0
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / total, 100. * correct / total
    
    def fit(self, train_loader, val_loader, num_epochs):
        """完整训练流程"""
        
        print("\n开始训练...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印结果
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        print("\n训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        path = os.path.join(self.config.save_dir, filename)
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'training_history.png'))
        plt.show()


# =============================================================================
# 5. 测试与评估
# =============================================================================

@torch.no_grad()
def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\n测试准确率: {accuracy:.2f}%")
    
    return accuracy, np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)
    
    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().squeeze()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'真实: {true_label}, 预测: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    """主函数"""
    # 配置
    config = Config()
    
    # 设置
    device = setup_device()
    set_seed(config.seed)
    
    # 数据
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 模型（可选择 MLP 或 CNN）
    # model = MLP(config.input_size, config.hidden_sizes, 
    #            config.num_classes, config.dropout)
    model = CNN(num_classes=config.num_classes, dropout=config.dropout)
    
    print(f"\n模型架构:")
    print(model)
    print(f"\n参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # 训练
    trainer = Trainer(model, criterion, optimizer, scheduler, device, config)
    history = trainer.fit(train_loader, val_loader, config.num_epochs)
    
    # 绘制训练历史
    trainer.plot_history()
    
    # 加载最佳模型并测试
    trainer.load_checkpoint('best_model.pth')
    test_acc, preds, targets = test_model(trainer.model, test_loader, device)
    
    # 可视化
    visualize_predictions(trainer.model, test_loader, device)
    plot_confusion_matrix(targets, preds)
    
    print("\n训练完成!")


def demo_quick_training():
    """快速演示（减少训练时间）"""
    print("=" * 60)
    print("MNIST 快速训练演示")
    print("=" * 60)
    
    device = setup_device()
    set_seed(42)
    
    # 简化的数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 只使用部分数据
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, transform=transform, download=True
    )
    
    # 取前 5000 个样本
    train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 简单模型
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 快速训练
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, "
              f"Acc = {100.*correct/total:.2f}%")
    
    print("\n快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_quick_training()
    else:
        main()

