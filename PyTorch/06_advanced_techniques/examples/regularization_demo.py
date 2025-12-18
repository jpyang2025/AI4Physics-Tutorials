#!/usr/bin/env python3
"""
正则化技术演示

本示例演示各种正则化技术的效果：
1. L1/L2 正则化
2. Dropout
3. Batch Normalization
4. 数据增强
5. 早停

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. 数据生成
# =============================================================================

def generate_classification_data(n_samples=1000, n_features=20, n_informative=5,
                                  noise=0.1, random_state=42):
    """
    生成分类数据
    
    只有前 n_informative 个特征是有用的
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # 只用前几个特征生成标签
    w = np.zeros(n_features)
    w[:n_informative] = np.random.randn(n_informative)
    
    logits = X @ w + noise * np.random.randn(n_samples)
    y = (logits > 0).astype(np.float32)
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    
    return X, y, w


def create_dataloaders(X, y, batch_size=32, val_ratio=0.2):
    """创建训练和验证数据加载器"""
    dataset = TensorDataset(X, y)
    
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


# =============================================================================
# 2. 模型定义
# =============================================================================

class SimpleMLP(nn.Module):
    """简单 MLP（无正则化）"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPWithDropout(nn.Module):
    """带 Dropout 的 MLP"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class MLPWithBatchNorm(nn.Module):
    """带 BatchNorm 的 MLP"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
# 3. 训练函数
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=100, l1_lambda=0, l2_lambda=0, device='cpu',
                early_stopping_patience=None):
    """
    训练模型
    
    Args:
        l1_lambda: L1 正则化系数
        l2_lambda: L2 正则化系数（手动实现，与 weight_decay 不同）
        early_stopping_patience: 早停耐心值
    """
    model = model.to(device)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # L1 正则化
            if l1_lambda > 0:
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_reg
            
            # L2 正则化（手动）
            if l2_lambda > 0:
                l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss /= total
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= total
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 早停
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停于 epoch {epoch}")
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break
    
    return history


# =============================================================================
# 4. 演示函数
# =============================================================================

def demo_l1_l2_regularization():
    """演示 L1 和 L2 正则化的效果"""
    print("\n" + "=" * 60)
    print("演示 1: L1 vs L2 正则化")
    print("=" * 60)
    
    # 生成数据
    X, y, true_weights = generate_classification_data(
        n_samples=500, n_features=50, n_informative=10
    )
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 不同正则化设置
    configs = [
        {'name': '无正则化', 'l1': 0, 'l2': 0, 'wd': 0},
        {'name': 'L2 (weight_decay)', 'l1': 0, 'l2': 0, 'wd': 0.01},
        {'name': 'L1', 'l1': 0.001, 'l2': 0, 'wd': 0},
        {'name': 'L1 + L2', 'l1': 0.0005, 'l2': 0.005, 'wd': 0},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        
        model = SimpleMLP(50, 32, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=config['wd'])
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, l1_lambda=config['l1'], l2_lambda=config['l2'],
            device=device
        )
        
        results[config['name']] = {
            'history': history,
            'weights': model.fc1.weight.data.cpu().numpy()
        }
        
        print(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 训练曲线
    for name, data in results.items():
        axes[0, 0].plot(data['history']['train_loss'], label=f'{name} (train)')
        axes[0, 1].plot(data['history']['val_loss'], label=f'{name} (val)')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练损失')
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('验证损失')
    axes[0, 1].legend()
    
    # 权重分布
    for i, (name, data) in enumerate(results.items()):
        weights = data['weights'].flatten()
        axes[1, 0].hist(weights, bins=50, alpha=0.5, label=name)
    
    axes[1, 0].set_xlabel('权重值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('权重分布')
    axes[1, 0].legend()
    
    # 权重稀疏性
    sparsity = []
    for name, data in results.items():
        weights = data['weights'].flatten()
        sparse_ratio = np.sum(np.abs(weights) < 0.01) / len(weights) * 100
        sparsity.append(sparse_ratio)
    
    axes[1, 1].bar(list(results.keys()), sparsity, color='steelblue')
    axes[1, 1].set_ylabel('稀疏比例 (%)')
    axes[1, 1].set_title('权重稀疏性 (|w| < 0.01)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('l1_l2_comparison.png', dpi=150)
    plt.show()


def demo_dropout():
    """演示 Dropout 的效果"""
    print("\n" + "=" * 60)
    print("演示 2: Dropout")
    print("=" * 60)
    
    # 生成数据（故意制造过拟合场景）
    X, y, _ = generate_classification_data(n_samples=200, n_features=50)
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dropout_rates = [0, 0.2, 0.5, 0.7]
    results = {}
    
    for dropout in dropout_rates:
        print(f"\nDropout = {dropout}")
        
        model = MLPWithDropout(50, 64, 2, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100, device=device
        )
        
        results[f'Dropout={dropout}'] = history
        print(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for name, history in results.items():
        axes[0].plot(history['train_acc'], label=f'{name} (train)', linestyle='--')
        axes[0].plot(history['val_acc'], label=f'{name} (val)')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Dropout 对过拟合的影响')
    axes[0].legend()
    
    # 过拟合程度（训练-验证差距）
    final_gaps = []
    for name, history in results.items():
        gap = history['train_acc'][-1] - history['val_acc'][-1]
        final_gaps.append(gap)
    
    axes[1].bar(list(results.keys()), final_gaps, color='coral')
    axes[1].set_ylabel('训练-验证准确率差距 (%)')
    axes[1].set_title('过拟合程度')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dropout_comparison.png', dpi=150)
    plt.show()


def demo_batchnorm():
    """演示 BatchNorm 的效果"""
    print("\n" + "=" * 60)
    print("演示 3: Batch Normalization")
    print("=" * 60)
    
    X, y, _ = generate_classification_data(n_samples=500, n_features=50)
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 比较有无 BatchNorm
    models = {
        '无 BatchNorm': SimpleMLP(50, 64, 2),
        '有 BatchNorm': MLPWithBatchNorm(50, 64, 2, dropout=0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练: {name}")
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, device=device
        )
        
        results[name] = history
        print(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for name, history in results.items():
        axes[0].plot(history['train_loss'], label=f'{name}')
        axes[1].plot(history['val_acc'], label=f'{name}')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失')
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('验证准确率')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('batchnorm_comparison.png', dpi=150)
    plt.show()


def demo_early_stopping():
    """演示早停的效果"""
    print("\n" + "=" * 60)
    print("演示 4: Early Stopping")
    print("=" * 60)
    
    # 生成容易过拟合的数据
    X, y, _ = generate_classification_data(n_samples=200, n_features=50)
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        {'name': '无早停', 'patience': None},
        {'name': '早停 (patience=10)', 'patience': 10},
        {'name': '早停 (patience=20)', 'patience': 20},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        
        model = SimpleMLP(50, 64, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=200, device=device,
            early_stopping_patience=config['patience']
        )
        
        results[config['name']] = history
        print(f"  训练 epoch 数: {len(history['train_loss'])}")
        print(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for name, history in results.items():
        epochs = range(len(history['val_loss']))
        axes[0].plot(epochs, history['val_loss'], label=name)
        axes[1].plot(epochs, history['val_acc'], label=name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('验证损失')
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('验证准确率')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=150)
    plt.show()


def demo_combined_regularization():
    """演示组合正则化的效果"""
    print("\n" + "=" * 60)
    print("演示 5: 组合正则化")
    print("=" * 60)
    
    X, y, _ = generate_classification_data(n_samples=300, n_features=50)
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 不同组合
    configs = [
        {'name': '基线', 'model': 'simple', 'wd': 0, 'patience': None},
        {'name': 'Dropout', 'model': 'dropout', 'wd': 0, 'patience': None},
        {'name': 'BatchNorm + Dropout', 'model': 'bn', 'wd': 0, 'patience': None},
        {'name': 'BN + Dropout + L2 + 早停', 'model': 'bn', 'wd': 0.01, 'patience': 15},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        
        if config['model'] == 'simple':
            model = SimpleMLP(50, 64, 2)
        elif config['model'] == 'dropout':
            model = MLPWithDropout(50, 64, 2, dropout=0.3)
        else:
            model = MLPWithBatchNorm(50, 64, 2, dropout=0.3)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=config['wd'])
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100, device=device,
            early_stopping_patience=config['patience']
        )
        
        results[config['name']] = history
        print(f"  训练 epoch 数: {len(history['train_loss'])}")
        print(f"  最佳验证准确率: {max(history['val_acc']):.2f}%")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, history in results.items():
        ax.plot(history['val_acc'], label=name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('不同正则化组合的效果比较')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_regularization.png', dpi=150)
    plt.show()
    
    # 最终结果汇总
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    print(f"{'配置':<30} {'最佳验证准确率':>15} {'最终验证准确率':>15}")
    print("-" * 60)
    for name, history in results.items():
        best = max(history['val_acc'])
        final = history['val_acc'][-1]
        print(f"{name:<30} {best:>14.2f}% {final:>14.2f}%")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有演示"""
    print("正则化技术演示")
    print("=" * 60)
    
    demo_l1_l2_regularization()
    demo_dropout()
    demo_batchnorm()
    demo_early_stopping()
    demo_combined_regularization()
    
    print("\n所有演示完成！")


def quick_demo():
    """快速演示"""
    print("正则化快速演示")
    print("=" * 60)
    
    # 生成数据
    X, y, _ = generate_classification_data(n_samples=300, n_features=30)
    train_loader, val_loader = create_dataloaders(X, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 简单比较
    models = {
        '无正则化': SimpleMLP(30, 32, 2),
        'Dropout=0.5': MLPWithDropout(30, 32, 2, dropout=0.5),
    }
    
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=30, device=device
        )
        
        print(f"{name}: 验证准确率 = {history['val_acc'][-1]:.2f}%")
    
    print("\n快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        main()

