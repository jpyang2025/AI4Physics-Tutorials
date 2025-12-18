# 6.1 正则化技术

## 📖 概述

正则化是防止模型过拟合的核心技术。从物理角度看，正则化相当于在损失函数中添加"约束势能"，限制模型参数的自由度。

## 🎯 学习目标

- 理解过拟合的本质
- 掌握 L1/L2 正则化
- 使用 Dropout 和 Batch Normalization
- 了解数据增强作为正则化

---

## 6.1.1 过拟合与欠拟合

### 什么是过拟合？

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_overfitting():
    """演示过拟合现象"""
    # 真实函数
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    # 生成带噪声的训练数据
    np.random.seed(42)
    x_train = np.random.uniform(0, 1, 20)
    y_train = true_function(x_train) + 0.3 * np.random.randn(20)
    
    # 转换为张量
    X = torch.from_numpy(x_train).float().reshape(-1, 1)
    Y = torch.from_numpy(y_train).float().reshape(-1, 1)
    
    # 创建多项式特征
    def polynomial_features(x, degree):
        return torch.cat([x**i for i in range(degree + 1)], dim=1)
    
    # 不同复杂度的模型
    degrees = [1, 4, 15]  # 欠拟合、适当、过拟合
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    for ax, degree in zip(axes, degrees):
        # 多项式特征
        X_poly = polynomial_features(X, degree)
        X_test_poly = polynomial_features(x_test, degree)
        
        # 线性回归
        model = nn.Linear(degree + 1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        for _ in range(1000):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_poly), Y)
            loss.backward()
            optimizer.step()
        
        # 预测
        with torch.no_grad():
            y_pred = model(X_test_poly)
        
        ax.scatter(x_train, y_train, c='blue', label='训练数据')
        ax.plot(x_test.numpy(), true_function(x_test.numpy()), 
                'g--', label='真实函数')
        ax.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='拟合结果')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'多项式阶数 = {degree}')
        ax.legend()
        ax.set_ylim(-2, 2)
    
    axes[0].set_title('欠拟合 (degree=1)')
    axes[1].set_title('适当拟合 (degree=4)')
    axes[2].set_title('过拟合 (degree=15)')
    
    plt.tight_layout()
    plt.show()
```

### 物理直觉：自由度与约束

从统计力学角度：

- **模型参数数**：系统自由度 $N_f$
- **训练样本数**：对系统的约束 $N_c$
- **有效自由度**：$N_{\text{eff}} = N_f - N_c$

当 $N_{\text{eff}} \gg 0$ 时，系统有太多未被约束的自由度，导致过拟合。

---

## 6.1.2 L2 正则化（权重衰减）

### 物理类比：谐振子势能

L2 正则化在损失函数中添加权重的平方和：

$$L_{\text{reg}} = L + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

这相当于给每个权重添加一个**谐振子势能**，倾向于使权重保持在原点附近。

### PyTorch 实现

```python
# 方法1：在优化器中设置 weight_decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 正则化系数
)

# 方法2：手动添加正则化项
def train_with_l2_regularization(model, train_loader, criterion, 
                                  optimizer, l2_lambda=1e-4):
    """带 L2 正则化的训练"""
    model.train()
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 添加 L2 正则化项
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)**2
        
        loss = loss + l2_lambda * l2_reg
        
        loss.backward()
        optimizer.step()
```

### L2 正则化的效果

```python
def compare_l2_regularization():
    """比较不同 L2 正则化强度的效果"""
    
    # 生成数据
    torch.manual_seed(42)
    X = torch.randn(100, 20)  # 100 个样本，20 个特征
    w_true = torch.zeros(20, 1)
    w_true[:5] = torch.randn(5, 1)  # 只有前5个特征有用
    Y = X @ w_true + 0.1 * torch.randn(100, 1)
    
    l2_lambdas = [0, 1e-3, 1e-2, 1e-1, 1.0]
    
    fig, axes = plt.subplots(1, len(l2_lambdas), figsize=(15, 3))
    
    for ax, l2_lambda in zip(axes, l2_lambdas):
        model = nn.Linear(20, 1, bias=False)
        
        # 使用 weight_decay 实现 L2 正则化
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, 
                                    weight_decay=l2_lambda)
        
        for _ in range(1000):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), Y)
            loss.backward()
            optimizer.step()
        
        # 可视化权重
        weights = model.weight.data.numpy().flatten()
        ax.bar(range(20), weights)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('特征索引')
        ax.set_ylabel('权重值')
        ax.set_title(f'λ = {l2_lambda}')
        ax.set_ylim(-1.5, 1.5)
    
    plt.suptitle('L2 正则化对权重的影响')
    plt.tight_layout()
    plt.show()
```

---

## 6.1.3 L1 正则化

### 物理类比：各向异性势能

L1 正则化添加权重的绝对值和：

$$L_{\text{reg}} = L + \lambda\|\mathbf{w}\|_1$$

这倾向于产生**稀疏解**（许多权重为零），类似于各向异性晶体势能。

### PyTorch 实现

```python
def train_with_l1_regularization(model, train_loader, criterion, 
                                  optimizer, l1_lambda=1e-4):
    """带 L1 正则化的训练"""
    model.train()
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 添加 L1 正则化项
        l1_reg = torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        
        loss = loss + l1_lambda * l1_reg
        
        loss.backward()
        optimizer.step()
```

### L1 vs L2：稀疏性比较

```python
def compare_l1_l2():
    """比较 L1 和 L2 正则化"""
    
    torch.manual_seed(42)
    X = torch.randn(100, 50)
    w_true = torch.zeros(50, 1)
    w_true[:10] = torch.randn(10, 1)  # 只有前10个特征有用
    Y = X @ w_true + 0.1 * torch.randn(100, 1)
    
    results = {}
    
    # 无正则化
    model_none = nn.Linear(50, 1, bias=False)
    opt_none = torch.optim.Adam(model_none.parameters(), lr=0.01)
    for _ in range(1000):
        opt_none.zero_grad()
        loss = nn.MSELoss()(model_none(X), Y)
        loss.backward()
        opt_none.step()
    results['无正则化'] = model_none.weight.data.numpy().flatten()
    
    # L2 正则化
    model_l2 = nn.Linear(50, 1, bias=False)
    opt_l2 = torch.optim.Adam(model_l2.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(1000):
        opt_l2.zero_grad()
        loss = nn.MSELoss()(model_l2(X), Y)
        loss.backward()
        opt_l2.step()
    results['L2'] = model_l2.weight.data.numpy().flatten()
    
    # L1 正则化
    model_l1 = nn.Linear(50, 1, bias=False)
    opt_l1 = torch.optim.Adam(model_l1.parameters(), lr=0.01)
    for _ in range(1000):
        opt_l1.zero_grad()
        loss = nn.MSELoss()(model_l1(X), Y)
        l1_reg = 0.1 * sum(p.abs().sum() for p in model_l1.parameters())
        (loss + l1_reg).backward()
        opt_l1.step()
    results['L1'] = model_l1.weight.data.numpy().flatten()
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, weights) in zip(axes, results.items()):
        ax.bar(range(50), weights, color='steelblue')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('特征索引')
        ax.set_ylabel('权重值')
        ax.set_title(f'{name}\n非零权重数: {np.sum(np.abs(weights) > 0.01)}')
    
    plt.tight_layout()
    plt.show()
```

---

## 6.1.4 Dropout

### 物理类比：随机稀释

Dropout 在训练时随机"关闭"一部分神经元，类似于**格点模型中的随机稀释**。

```python
import torch.nn.functional as F

class NetworkWithDropout(nn.Module):
    """带 Dropout 的网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 训练时随机丢弃
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Dropout 的工作原理
def dropout_demonstration():
    """演示 Dropout 的工作原理"""
    
    dropout = nn.Dropout(p=0.5)
    
    x = torch.ones(1, 10)
    
    # 训练模式
    dropout.train()
    print("训练模式 (随机丢弃):")
    for i in range(3):
        out = dropout(x)
        print(f"  尝试 {i+1}: {out}")
    
    # 评估模式
    dropout.eval()
    print("\n评估模式 (无丢弃):")
    out = dropout(x)
    print(f"  输出: {out}")
```

### Dropout 变体

```python
# 标准 Dropout
dropout = nn.Dropout(p=0.5)

# 2D Dropout（用于 CNN）
dropout2d = nn.Dropout2d(p=0.5)  # 丢弃整个通道

# Alpha Dropout（用于 SELU 激活）
alpha_dropout = nn.AlphaDropout(p=0.5)

# 使用示例
class CNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout2d = nn.Dropout2d(0.25)  # 卷积层后使用 2D dropout
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)       # 全连接层后使用标准 dropout
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Dropout 的物理意义

从统计力学角度，Dropout 可以理解为：

1. **系综平均**：每次前向传播使用不同的子网络，最终结果是对所有可能子网络的平均
2. **噪声注入**：类似于有限温度下的热涨落
3. **模型集成**：隐式地训练了 $2^N$ 个子模型（N 是神经元数）

---

## 6.1.5 Batch Normalization

### 物理类比：重整化

Batch Normalization 将每层的激活值标准化，类似于**重整化群变换**。

```python
class NetworkWithBatchNorm(nn.Module):
    """带 Batch Normalization 的网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 批归一化
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)      # 归一化
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x


# BatchNorm 的工作原理
def batchnorm_demonstration():
    """演示 BatchNorm 的工作原理"""
    
    bn = nn.BatchNorm1d(5)
    
    # 训练模式
    bn.train()
    x = torch.randn(32, 5) * 10 + 5  # 均值约5，标准差约10
    
    print(f"输入统计: 均值={x.mean().item():.2f}, 标准差={x.std().item():.2f}")
    
    y = bn(x)
    print(f"输出统计: 均值={y.mean().item():.2f}, 标准差={y.std().item():.2f}")
    
    # 评估模式使用运行时统计量
    bn.eval()
    x_test = torch.randn(8, 5) * 10 + 5
    y_test = bn(x_test)
    print(f"\n评估模式输出: 均值={y_test.mean().item():.2f}, 标准差={y_test.std().item():.2f}")
```

### 归一化变体

```python
# 不同的归一化方法
batch_norm = nn.BatchNorm1d(num_features)   # 对 batch 维度归一化
layer_norm = nn.LayerNorm(normalized_shape)  # 对特征维度归一化
instance_norm = nn.InstanceNorm2d(num_features)  # 对每个样本的空间维度归一化
group_norm = nn.GroupNorm(num_groups, num_channels)  # 对通道分组归一化

# 使用场景
# - BatchNorm: CNN、大批量训练
# - LayerNorm: Transformer、RNN、小批量
# - InstanceNorm: 风格迁移
# - GroupNorm: 小批量 CNN
```

### Layer Normalization（适合 Transformer）

```python
class TransformerBlock(nn.Module):
    """Transformer 块使用 LayerNorm"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力 + 残差 + LayerNorm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈 + 残差 + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
```

---

## 6.1.6 数据增强

### 物理类比：对称性约束

数据增强利用问题的**对称性**来扩充数据，类似于物理中利用对称性减少问题复杂度。

```python
import torchvision.transforms as T

# 图像数据增强
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.RandomCrop(32, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 强数据增强（如 RandAugment 风格）
strong_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 物理数据增强

```python
class PhysicsDataAugmentation:
    """物理数据的增强策略"""
    
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, x, sigma=0.1):
        """添加高斯噪声 - 模拟测量误差"""
        return x + sigma * torch.randn_like(x)
    
    def scale_invariance(self, x, scale_range=(0.8, 1.2)):
        """缩放不变性 - 物理量的量纲变换"""
        scale = torch.empty(1).uniform_(*scale_range).item()
        return x * scale
    
    def time_reversal(self, trajectory):
        """时间反演 - 对于可逆系统"""
        return torch.flip(trajectory, dims=[0])
    
    def rotation(self, coords, angle=None):
        """
        旋转变换 - 利用旋转对称性
        coords: [N, 2] 或 [N, 3]
        """
        if angle is None:
            angle = torch.empty(1).uniform_(0, 2 * np.pi).item()
        
        if coords.shape[1] == 2:
            # 2D 旋转
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
            return coords @ R.T
        else:
            # 3D 旋转（绕 z 轴）
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            return coords @ R.T
    
    def translation(self, coords, max_shift=0.1):
        """平移变换 - 利用平移对称性"""
        shift = torch.empty(coords.shape[1]).uniform_(-max_shift, max_shift)
        return coords + shift


# 使用示例
augmenter = PhysicsDataAugmentation()

# 粒子坐标增强
coords = torch.randn(10, 3)  # 10个粒子的3D坐标
coords_aug = augmenter.rotation(coords)
coords_aug = augmenter.add_gaussian_noise(coords_aug, sigma=0.05)
```

---

## 6.1.7 其他正则化技术

### 早停（Early Stopping）

```python
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() 
                                     for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.should_stop
```

### 标签平滑（Label Smoothing）

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        
        # 创建平滑标签
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵
        log_probs = F.log_softmax(pred, dim=-1)
        loss = (-smooth_target * log_probs).sum(dim=-1).mean()
        
        return loss


# 使用
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### Mixup 数据增强

```python
def mixup_data(x, y, alpha=0.2):
    """
    Mixup 数据增强
    
    将两个样本线性混合
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 训练循环中使用
for inputs, targets in train_loader:
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
    
    outputs = model(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 6.1.8 正则化强度选择

### 超参数搜索

```python
from sklearn.model_selection import ParameterGrid

def regularization_search(model_fn, train_loader, val_loader, device):
    """网格搜索正则化超参数"""
    
    param_grid = {
        'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2],
        'dropout': [0, 0.1, 0.2, 0.3, 0.5]
    }
    
    best_val_loss = float('inf')
    best_params = None
    
    for params in ParameterGrid(param_grid):
        print(f"测试参数: {params}")
        
        # 创建模型
        model = model_fn(dropout=params['dropout']).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay=params['weight_decay']
        )
        
        # 训练
        for epoch in range(20):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += F.cross_entropy(model(inputs), targets).item()
        val_loss /= len(val_loader)
        
        print(f"  验证损失: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
    
    print(f"\n最佳参数: {best_params}")
    return best_params
```

---

## 🔬 物理视角总结

### 正则化的统计力学解释

| 正则化 | 物理对应 | 效果 |
|--------|---------|------|
| L2 | 谐振子势 | 权重趋向小值 |
| L1 | 各向异性势 | 产生稀疏解 |
| Dropout | 随机稀释 | 集成效应 |
| BatchNorm | 重整化 | 稳定训练 |
| 早停 | 有限时间 | 避免过度弛豫 |

### 选择指南

| 场景 | 推荐正则化 |
|------|-----------|
| 小数据集 | 强正则化（高 dropout、高 weight decay） |
| 大数据集 | 轻正则化 + 数据增强 |
| CNN | Dropout2D + BatchNorm + 数据增强 |
| Transformer | LayerNorm + Dropout + Label Smoothing |
| 物理问题 | 利用对称性的数据增强 |

---

## 📝 练习

1. 比较 L1 和 L2 正则化在特征选择上的效果
2. 实现 Mixup 数据增强并观察其效果
3. 为你的物理数据设计合适的数据增强策略

---

## ⏭️ 下一节

下一节我们将学习 [学习率调度](./02_learning_rate_scheduling.md)，了解如何动态调整学习率以获得更好的训练效果。

