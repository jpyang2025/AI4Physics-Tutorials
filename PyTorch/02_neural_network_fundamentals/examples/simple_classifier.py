#!/usr/bin/env python3
"""
简单分类器示例

本脚本演示如何使用 PyTorch 实现各种分类器，
包括逻辑回归、多层感知机和多分类问题。

适合有 Python 基础的物理科研人员学习。

运行方式：
    python simple_classifier.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import Tuple


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 二分类逻辑回归
# ============================================================

def generate_binary_data(n_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成二分类数据（两个高斯簇）"""
    torch.manual_seed(42)
    
    n_per_class = n_samples // 2
    
    # 类别 0: 中心在 (-1.5, -1.5)
    X0 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([-1.5, -1.5])
    y0 = torch.zeros(n_per_class)
    
    # 类别 1: 中心在 (1.5, 1.5)
    X1 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([1.5, 1.5])
    y1 = torch.ones(n_per_class)
    
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    # 打乱数据
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


class LogisticRegression(nn.Module):
    """逻辑回归模型"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def binary_classification_demo():
    """二分类逻辑回归演示"""
    print_section("1. 二分类逻辑回归")
    
    # 生成数据
    X, y = generate_binary_data(200)
    y = y.reshape(-1, 1)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: 类0={int((y==0).sum())}, 类1={int((y==1).sum())}")
    
    # 创建模型
    model = LogisticRegression(input_dim=2)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    
    # 训练
    epochs = 100
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(X)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            accuracy = ((y_pred > 0.5) == y).float().mean()
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, accuracy={accuracy.item():.2%}")
    
    # 最终结果
    y_pred_final = model(X)
    accuracy_final = ((y_pred_final > 0.5) == y).float().mean()
    print(f"\n最终准确率: {accuracy_final.item():.2%}")
    
    # 决策边界参数
    w = model.linear.weight.data.flatten()
    b = model.linear.bias.item()
    print(f"决策边界: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f} = 0")
    
    return X, y, model, losses


# ============================================================
# 2. XOR 问题与多层感知机
# ============================================================

class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


def xor_problem_demo():
    """XOR 问题演示：线性不可分问题"""
    print_section("2. XOR 问题（多层感知机）")
    
    # XOR 数据
    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    
    print("XOR 真值表:")
    print("  x1  x2  |  y")
    print("  --------|----")
    for i in range(4):
        print(f"   {int(X[i,0])}   {int(X[i,1])}  |  {int(y[i])}")
    
    # 线性模型（会失败）
    print("\n尝试线性模型...")
    linear_model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    
    optimizer = optim.Adam(linear_model.parameters(), lr=0.5)
    criterion = nn.BCELoss()
    
    for _ in range(1000):
        y_pred = linear_model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("线性模型预测:")
    for i in range(4):
        pred = linear_model(X[i:i+1]).item()
        print(f"  ({int(X[i,0])}, {int(X[i,1])}) -> {pred:.4f} (期望: {int(y[i])})")
    
    # 多层感知机（会成功）
    print("\n使用多层感知机...")
    mlp = MLP(input_dim=2, hidden_dim=4, output_dim=1)
    
    optimizer = optim.Adam(mlp.parameters(), lr=0.1)
    
    for epoch in range(2000):
        y_pred = mlp(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("MLP 预测:")
    for i in range(4):
        pred = mlp(X[i:i+1]).item()
        correct = "✓" if (pred > 0.5) == y[i].item() else "✗"
        print(f"  ({int(X[i,0])}, {int(X[i,1])}) -> {pred:.4f} (期望: {int(y[i])}) {correct}")
    
    return mlp


# ============================================================
# 3. 多分类问题
# ============================================================

def generate_multiclass_data(n_samples: int = 300, n_classes: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成多分类数据（多个高斯簇）"""
    torch.manual_seed(42)
    
    n_per_class = n_samples // n_classes
    
    # 在圆周上均匀分布类中心
    centers = []
    radius = 3.0
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    X_list, y_list = [], []
    for i, center in enumerate(centers):
        X_i = torch.randn(n_per_class, 2) * 0.7 + torch.tensor(center)
        y_i = torch.full((n_per_class,), i, dtype=torch.long)
        X_list.append(X_i)
        y_list.append(y_i)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # 打乱
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


class MultiClassMLP(nn.Module):
    """多分类 MLP"""
    
    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        # 注意：不加 Softmax，因为 CrossEntropyLoss 会处理
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def multiclass_classification_demo():
    """多分类问题演示"""
    print_section("3. 多分类问题")
    
    # 生成数据
    n_classes = 4
    X, y = generate_multiclass_data(n_samples=400, n_classes=n_classes)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别数量: {n_classes}")
    for c in range(n_classes):
        print(f"  类别 {c}: {(y == c).sum().item()} 个样本")
    
    # 划分训练集和测试集
    n_train = int(0.8 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 创建模型
    model = MultiClassMLP(input_dim=2, hidden_dims=[32, 16], n_classes=n_classes)
    print(f"\n模型结构:\n{model}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    epochs = 200
    train_losses, test_accuracies = [], []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 评估
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_pred = test_logits.argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()
            test_accuracies.append(test_acc)
        
        if epoch % 40 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, test_acc={test_acc:.2%}")
    
    print(f"\n最终测试准确率: {test_accuracies[-1]:.2%}")
    
    # 混淆矩阵
    with torch.no_grad():
        test_pred = model(X_test).argmax(dim=1)
    
    print("\n混淆矩阵:")
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y_test, test_pred):
        confusion[t, p] += 1
    
    print("     预测")
    print("      " + "  ".join([f"{i}" for i in range(n_classes)]))
    for i in range(n_classes):
        row = "  ".join([f"{confusion[i, j].item():2d}" for j in range(n_classes)])
        print(f"真{i}:  {row}")
    
    return X, y, model, X_train, y_train, X_test, y_test


# ============================================================
# 4. 非线性决策边界（螺旋数据）
# ============================================================

def generate_spiral_data(n_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成螺旋数据（强非线性）"""
    torch.manual_seed(42)
    
    n = n_samples // 2
    
    # 螺旋 1
    theta1 = torch.linspace(0, 4*np.pi, n)
    r1 = torch.linspace(0.5, 3, n)
    x1 = r1 * torch.cos(theta1) + torch.randn(n) * 0.1
    y1 = r1 * torch.sin(theta1) + torch.randn(n) * 0.1
    
    # 螺旋 2 (旋转 π)
    theta2 = torch.linspace(0, 4*np.pi, n) + np.pi
    r2 = torch.linspace(0.5, 3, n)
    x2 = r2 * torch.cos(theta2) + torch.randn(n) * 0.1
    y2 = r2 * torch.sin(theta2) + torch.randn(n) * 0.1
    
    X = torch.stack([
        torch.cat([x1, x2]),
        torch.cat([y1, y2])
    ], dim=1)
    
    y = torch.cat([torch.zeros(n), torch.ones(n)])
    
    # 打乱
    perm = torch.randperm(len(y))
    return X[perm], y[perm].long()


def spiral_classification_demo():
    """螺旋分类问题演示"""
    print_section("4. 螺旋分类（复杂非线性边界）")
    
    X, y = generate_spiral_data(300)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print("这是一个经典的非线性分类问题，两个类别呈螺旋状交织")
    
    # 深层网络
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    print(f"\n使用 4 层 MLP...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    for epoch in range(1000):
        logits = model(X)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            print(f"Epoch {epoch:4d}: loss={loss.item():.4f}, accuracy={acc.item():.2%}")
    
    # 最终准确率
    with torch.no_grad():
        final_pred = model(X).argmax(dim=1)
        final_acc = (final_pred == y).float().mean()
    
    print(f"\n最终准确率: {final_acc.item():.2%}")
    
    return X, y, model


# ============================================================
# 5. 物理应用：粒子分类
# ============================================================

def particle_classification_demo():
    """物理应用：根据运动学特征分类粒子"""
    print_section("5. 物理应用：粒子分类")
    
    torch.manual_seed(42)
    
    # 模拟粒子物理数据
    # 特征：[px, py, pz, E]（动量分量和能量）
    # 类别：0=电子, 1=缪子, 2=质子
    
    n_per_class = 200
    
    # 电子（轻，能量较低）
    m_e = 0.511  # MeV
    p_e = torch.randn(n_per_class, 3) * 0.3 + 0.5
    E_e = torch.sqrt((p_e ** 2).sum(dim=1, keepdim=True) + m_e**2)
    X_e = torch.cat([p_e, E_e], dim=1)
    
    # 缪子（中等质量）
    m_mu = 105.7  # MeV
    p_mu = torch.randn(n_per_class, 3) * 0.5 + 1.0
    E_mu = torch.sqrt((p_mu ** 2).sum(dim=1, keepdim=True) + m_mu**2)
    X_mu = torch.cat([p_mu, E_mu], dim=1)
    
    # 质子（重）
    m_p = 938.3  # MeV
    p_p = torch.randn(n_per_class, 3) * 0.7 + 1.5
    E_p = torch.sqrt((p_p ** 2).sum(dim=1, keepdim=True) + m_p**2)
    X_p = torch.cat([p_p, E_p], dim=1)
    
    # 合并数据
    X = torch.cat([X_e, X_mu, X_p], dim=0)
    y = torch.cat([
        torch.zeros(n_per_class),
        torch.ones(n_per_class),
        torch.full((n_per_class,), 2)
    ]).long()
    
    # 特征标准化（物理中常用）
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X_norm = (X - X_mean) / X_std
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征: [px, py, pz, E]")
    print(f"类别: 0=电子, 1=缪子, 2=质子")
    print(f"\n各类别能量统计:")
    print(f"  电子: E_mean={E_e.mean().item():.2f} MeV")
    print(f"  缪子: E_mean={E_mu.mean().item():.2f} MeV")
    print(f"  质子: E_mean={E_p.mean().item():.2f} MeV")
    
    # 打乱并划分数据
    perm = torch.randperm(len(y))
    X_norm, y = X_norm[perm], y[perm]
    
    n_train = int(0.8 * len(y))
    X_train, X_test = X_norm[:n_train], X_norm[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 模型
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    print("\n训练粒子分类器...")
    for epoch in range(300):
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 60 == 0:
            with torch.no_grad():
                test_pred = model(X_test).argmax(dim=1)
                test_acc = (test_pred == y_test).float().mean()
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, test_acc={test_acc:.2%}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = test_logits.argmax(dim=1)
        test_acc = (test_pred == y_test).float().mean()
    
    print(f"\n最终测试准确率: {test_acc:.2%}")
    
    # 每类准确率
    print("\n每类准确率:")
    particle_names = ['电子', '缪子', '质子']
    for c in range(3):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = (test_pred[mask] == y_test[mask]).float().mean()
            print(f"  {particle_names[c]}: {class_acc:.2%}")
    
    return model


# ============================================================
# 6. 可视化
# ============================================================

def visualize_classifiers(X_binary, y_binary, model_binary, losses_binary,
                          X_multi, y_multi, model_multi, 
                          X_spiral, y_spiral, model_spiral):
    """可视化分类结果"""
    print_section("6. 生成可视化图表")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 图1：二分类数据和决策边界
    ax = axes[0, 0]
    colors = ['blue' if yi == 0 else 'red' for yi in y_binary]
    ax.scatter(X_binary[:, 0].numpy(), X_binary[:, 1].numpy(), c=colors, alpha=0.5)
    
    # 画决策边界
    w = model_binary.linear.weight.data.flatten()
    b = model_binary.linear.bias.item()
    x_line = np.linspace(-4, 4, 100)
    y_line = -(w[0].item() * x_line + b) / w[1].item()
    ax.plot(x_line, y_line, 'k-', linewidth=2, label='决策边界')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('二分类逻辑回归')
    ax.legend()
    
    # 图2：二分类损失曲线
    ax = axes[0, 1]
    ax.plot(losses_binary)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('训练损失曲线')
    ax.grid(True, alpha=0.3)
    
    # 图3：多分类数据
    ax = axes[0, 2]
    n_classes = y_multi.max().item() + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for c in range(n_classes):
        mask = y_multi == c
        ax.scatter(X_multi[mask, 0].numpy(), X_multi[mask, 1].numpy(), 
                  c=[colors[c]], label=f'类别 {c}', alpha=0.6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('多分类问题')
    ax.legend()
    
    # 图4：多分类决策区域
    ax = axes[1, 0]
    xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model_multi(grid).argmax(dim=1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, levels=n_classes-1)
    for c in range(n_classes):
        mask = y_multi == c
        ax.scatter(X_multi[mask, 0].numpy(), X_multi[mask, 1].numpy(), 
                  c=[colors[c]], s=20, alpha=0.6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('多分类决策区域')
    
    # 图5：螺旋数据
    ax = axes[1, 1]
    colors_spiral = ['blue' if yi == 0 else 'red' for yi in y_spiral]
    ax.scatter(X_spiral[:, 0].numpy(), X_spiral[:, 1].numpy(), c=colors_spiral, alpha=0.5, s=10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('螺旋分类数据')
    
    # 图6：螺旋决策区域
    ax = axes[1, 2]
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model_spiral(grid).argmax(dim=1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_spiral[:, 0].numpy(), X_spiral[:, 1].numpy(), c=colors_spiral, s=10, alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('螺旋分类决策区域')
    
    plt.tight_layout()
    plt.savefig('classifier_results.png', dpi=150)
    print("图表已保存为 classifier_results.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" PyTorch 分类器完整示例")
    print("="*60)
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 运行所有示例
    X_binary, y_binary, model_binary, losses_binary = binary_classification_demo()
    xor_problem_demo()
    X_multi, y_multi, model_multi, _, _, _, _ = multiclass_classification_demo()
    X_spiral, y_spiral, model_spiral = spiral_classification_demo()
    particle_classification_demo()
    
    # 可视化
    visualize_classifiers(
        X_binary, y_binary, model_binary, losses_binary,
        X_multi, y_multi, model_multi,
        X_spiral, y_spiral, model_spiral
    )
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 二分类使用 BCELoss + Sigmoid")
    print("  2. 多分类使用 CrossEntropyLoss（内含 Softmax）")
    print("  3. 线性模型无法处理 XOR 等非线性问题")
    print("  4. 多层网络可以学习复杂的非线性决策边界")
    print("  5. 深层网络可以处理螺旋等高度非线性的分类问题")
    print()


if __name__ == "__main__":
    main()

