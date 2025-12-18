# 2.3 前馈神经网络

## 什么是前馈神经网络？

**前馈神经网络**（Feedforward Neural Network, FNN），也称为**多层感知机**（Multi-Layer Perceptron, MLP），是最基本的神经网络架构。

**"前馈"**意味着信息只沿一个方向流动：从输入层 → 隐藏层 → 输出层，没有循环或反馈连接。

## 网络结构

一个典型的前馈网络包含：

1. **输入层**：接收原始数据，维度由数据决定
2. **隐藏层**：一个或多个，进行特征变换
3. **输出层**：产生最终预测

### 数学表示

设网络有 $L$ 层，第 $l$ 层的输出为 $\mathbf{h}^{(l)}$：

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{h}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

其中：
- $\mathbf{W}^{(l)}$ 是第 $l$ 层的权重矩阵
- $\mathbf{b}^{(l)}$ 是偏置向量
- $\sigma$ 是激活函数
- $\mathbf{h}^{(0)} = \mathbf{x}$ 是输入

整个网络可以写成复合函数：

$$f(\mathbf{x}) = \sigma_L(\mathbf{W}^{(L)} \sigma_{L-1}(\mathbf{W}^{(L-1)} \cdots \sigma_1(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) \cdots + \mathbf{b}^{(L-1)}) + \mathbf{b}^{(L)})$$

## 万能近似定理

**万能近似定理**（Universal Approximation Theorem）是神经网络理论的基石：

> 一个具有单个隐藏层、足够多神经元和适当激活函数的前馈网络，可以以任意精度逼近任何连续函数。

### 数学表述

设 $\sigma$ 是一个非常数、有界、单调递增的连续函数。对于任何连续函数 $f: [0,1]^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在整数 $N$、实数 $v_i, b_i$ 和向量 $\mathbf{w}_i$，使得：

$$\left| f(\mathbf{x}) - \sum_{i=1}^{N} v_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i) \right| < \epsilon$$

对所有 $\mathbf{x} \in [0,1]^n$ 成立。

### 物理直觉

从物理角度理解万能近似定理：

1. **傅里叶级数类比**：傅里叶级数用正弦/余弦函数逼近周期函数，神经网络用激活函数作为"基函数"。

2. **基函数展开**：
   - 傅里叶：$f(x) = \sum_n a_n \sin(nx) + b_n \cos(nx)$
   - 神经网络：$f(\mathbf{x}) = \sum_i v_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)$
   
3. **每个隐藏神经元**就像一个"基函数"，由权重控制其位置和方向，偏置控制其位移。

### 演示：用神经网络拟合任意函数

```python
import torch
import torch.nn as nn
import numpy as np

# 目标函数：复杂的非线性函数
def target_function(x):
    return torch.sin(3 * x) + 0.5 * torch.cos(5 * x) + 0.3 * x

# 生成数据
x = torch.linspace(-3, 3, 200).reshape(-1, 1)
y = target_function(x)

# 不同宽度的网络
for width in [5, 20, 100]:
    model = nn.Sequential(
        nn.Linear(1, width),
        nn.Tanh(),
        nn.Linear(width, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    for epoch in range(2000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"隐藏层神经元数={width:3d}, 最终损失={loss.item():.6f}")

# 更多神经元 → 更好的逼近能力
```

## PyTorch 实现前馈网络

### 方法1：使用 nn.Sequential

最简单的方式，适合简单的顺序结构：

```python
import torch.nn as nn

# 一个三层网络：输入10维 -> 隐藏64维 -> 隐藏32维 -> 输出5维
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5)
)

# 使用
x = torch.randn(8, 10)  # 8个样本，每个10维
y = model(x)            # 输出形状 [8, 5]
print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
```

### 方法2：继承 nn.Module

更灵活的方式，可以定义复杂的前向传播逻辑：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        """
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如 [64, 32]
            output_dim: 输出维度
            activation: 激活函数类型
        """
        super().__init__()
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = MLP(
    input_dim=10,
    hidden_dims=[64, 128, 64],
    output_dim=5,
    activation='relu'
)

print(model)

# 测试
x = torch.randn(8, 10)
y = model(x)
print(f"输出形状: {y.shape}")
```

### 方法3：更灵活的自定义

```python
class FlexibleMLP(nn.Module):
    """带有残差连接和归一化的 MLP"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 隐藏层（带残差连接）
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # 输入投影
        x = self.input_layer(x)
        x = self.activation(x)
        
        # 隐藏层（带残差）
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = x + residual  # 残差连接
            x = norm(x)       # 层归一化
        
        # 输出
        return self.output_layer(x)

model = FlexibleMLP(input_dim=10, hidden_dim=64, output_dim=5, num_layers=4)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
```

## 网络深度 vs 宽度

### 深度的作用

深层网络可以学习**层次化特征**：

```
输入层 → 底层特征 → 中层特征 → 高层特征 → 输出
         (边缘)    (纹理)     (部件)     (类别)
```

**物理类比**：类似于重整化群，每一层都在不同尺度上提取特征。

### 宽度的作用

宽度决定了每一层能表示多少特征。根据万能近似定理，足够宽的单层网络理论上可以逼近任何函数，但：

- 所需神经元数量可能指数级增长
- 训练难度大

### 深度 vs 宽度的权衡

| 特性 | 深而窄 | 浅而宽 |
|------|--------|--------|
| 参数效率 | 高（指数级表示能力） | 低 |
| 训练难度 | 梯度消失/爆炸风险 | 相对容易 |
| 特征学习 | 层次化特征 | 扁平化特征 |
| 泛化能力 | 通常更好 | 容易过拟合 |

```python
# 对比实验：相近参数量，不同深度
import torch
import torch.nn as nn

# 浅而宽：1个隐藏层，512个神经元
shallow = nn.Sequential(
    nn.Linear(10, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)

# 深而窄：4个隐藏层，每层64个神经元
deep = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

print(f"浅网络参数: {sum(p.numel() for p in shallow.parameters())}")  # ~5.6k
print(f"深网络参数: {sum(p.numel() for p in deep.parameters())}")      # ~13k
```

## 参数初始化

良好的参数初始化对训练至关重要。

### 问题：不当初始化

```python
# 全零初始化的问题
linear = nn.Linear(100, 100)
nn.init.zeros_(linear.weight)

x = torch.randn(10, 100)
y = linear(x)
print(f"输出方差: {y.var().item():.6f}")  # 极小

# 所有神经元输出相同 → 梯度相同 → 对称性无法打破！
```

### Xavier/Glorot 初始化

适用于 Sigmoid 和 Tanh：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$

或正态分布版本：

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}+n_{out}}\right)$$

**目标**：保持前向传播和反向传播时信号的方差稳定。

```python
# Xavier 初始化
linear = nn.Linear(100, 100)
nn.init.xavier_uniform_(linear.weight)  # 均匀分布
# 或
nn.init.xavier_normal_(linear.weight)   # 正态分布
```

### Kaiming/He 初始化

专为 ReLU 设计（因为 ReLU 会将一半激活值置零）：

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

```python
# Kaiming 初始化
linear = nn.Linear(100, 100)
nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
# 或
nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
```

### PyTorch 默认初始化

`nn.Linear` 默认使用 Kaiming 均匀初始化：

```python
linear = nn.Linear(100, 100)
print(f"默认权重标准差: {linear.weight.std().item():.4f}")
# 大约 1/sqrt(100) = 0.1
```

## 模型分析工具

### 查看模型结构

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 打印模型
print(model)

# 详细参数统计
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"总参数: {total:,}")
print(f"可训练参数: {trainable:,}")
```

### 查看各层参数

```python
for name, param in model.named_parameters():
    print(f"{name:20} 形状: {str(list(param.shape)):15} 参数量: {param.numel()}")
```

### 使用 torchinfo 可视化

```python
# pip install torchinfo
from torchinfo import summary

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

summary(model, input_size=(32, 784))
```

## 前向传播的物理视角

### 信息流与能量流

前向传播可以看作信息在网络中的"流动"：

1. **信息变换**：每一层都在进行特征空间的变换
2. **维度变化**：降维类似于物理中的粗粒化
3. **非线性**：激活函数引入"非线性响应"

### 几何视角

神经网络在做**流形学习**：

- 输入数据位于高维空间的低维流形上
- 网络学习将这个流形"展开"到易于分类的空间
- 每一层都在"折叠"或"展开"空间

```python
# 演示：网络如何变换空间
import torch
import torch.nn as nn

# 简单的2D分类问题
# 两类数据呈同心圆分布（线性不可分）
torch.manual_seed(42)
n_points = 500

# 内圈（类0）
r_inner = torch.randn(n_points // 2) * 0.2 + 0.5
theta_inner = torch.rand(n_points // 2) * 2 * 3.14159
X_inner = torch.stack([r_inner * torch.cos(theta_inner), 
                       r_inner * torch.sin(theta_inner)], dim=1)
y_inner = torch.zeros(n_points // 2)

# 外圈（类1）
r_outer = torch.randn(n_points // 2) * 0.2 + 1.5
theta_outer = torch.rand(n_points // 2) * 2 * 3.14159
X_outer = torch.stack([r_outer * torch.cos(theta_outer), 
                       r_outer * torch.sin(theta_outer)], dim=1)
y_outer = torch.ones(n_points // 2)

X = torch.cat([X_inner, X_outer], dim=0)
y = torch.cat([y_inner, y_outer], dim=0).long()

# 网络
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

# 训练...
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(500):
    logits = model(X)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 准确率
pred = logits.argmax(dim=1)
accuracy = (pred == y).float().mean()
print(f"准确率: {accuracy.item():.2%}")
```

## 本节小结

| 概念 | 说明 |
|------|------|
| 前馈网络 | 信息单向流动的神经网络 |
| 万能近似定理 | 足够宽的单隐层网络可逼近任意连续函数 |
| 深度 vs 宽度 | 深度提供层次化特征，宽度提供表示容量 |
| 参数初始化 | Xavier（Sigmoid/Tanh）、Kaiming（ReLU） |
| `nn.Sequential` | 顺序堆叠层的简单方式 |
| `nn.Module` | 自定义网络的基类 |

## 练习

1. 实现一个能拟合 $f(x) = \sin(x) \cdot e^{-0.1x^2}$ 的网络
2. 比较不同深度（2层、4层、8层）的网络拟合同一函数的效果
3. 实现一个带 Dropout 的 MLP，观察其对过拟合的影响

## 延伸阅读

- Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function"
- Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks"
- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

---

[← 上一节：激活函数](./02_activation_functions.md) | [下一节：损失函数 →](./04_loss_functions.md)

