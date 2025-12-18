# 2.1 感知机与线性模型

## 从线性回归开始

在物理学中，我们经常需要从实验数据中拟合函数关系。最简单的情况是**线性回归**——寻找数据点之间的线性关系。

### 问题设定

给定数据集 $\{(x_i, y_i)\}_{i=1}^{N}$，我们希望找到参数 $w$ 和 $b$，使得：

$$\hat{y} = wx + b$$

能够最好地拟合数据。

### 损失函数：均方误差 (MSE)

"最好地拟合"需要定量定义。我们使用**均方误差**作为损失函数：

$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - wx_i - b)^2$$

**物理类比**：这就像弹簧系统的势能，每个数据点都像一个弹簧，将预测值"拉向"真实值。

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 生成模拟数据：y = 2x + 1 + 噪声
torch.manual_seed(42)
X = torch.linspace(-5, 5, 100).reshape(-1, 1)  # 形状 [100, 1]
y_true = 2 * X + 1 + torch.randn_like(X) * 0.5  # 真实斜率=2，截距=1

# 定义模型参数
w = torch.tensor([0.0], requires_grad=True)  # 权重
b = torch.tensor([0.0], requires_grad=True)  # 偏置

# 学习率
lr = 0.01

# 训练循环
for epoch in range(100):
    # 前向传播
    y_pred = w * X + b
    
    # 计算损失
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数（不追踪梯度）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n最终结果: y = {w.item():.4f}x + {b.item():.4f}")
print(f"真实关系: y = 2.0000x + 1.0000")
```

## 多元线性回归

当输入有多个特征时，模型变为：

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = \mathbf{w}^T \mathbf{x} + b$$

用矩阵形式表示：

$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times d}$ 是数据矩阵，$\mathbf{w} \in \mathbb{R}^d$ 是权重向量。

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 使用 nn.Linear 实现多元线性回归
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入维度 -> 1维输出
    
    def forward(self, x):
        return self.linear(x)

# 生成多维数据
# 真实关系: y = 2*x1 + 3*x2 - 1*x3 + 0.5
torch.manual_seed(42)
N, d = 200, 3
X = torch.randn(N, d)
w_true = torch.tensor([[2.0], [3.0], [-1.0]])
b_true = 0.5
y_true = X @ w_true + b_true + torch.randn(N, 1) * 0.3

# 创建模型
model = LinearRegression(input_dim=d)

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 损失函数
criterion = nn.MSELoss()

# 训练
for epoch in range(100):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 查看学习到的参数
print(f"\n学习到的权重: {model.linear.weight.data.flatten().tolist()}")
print(f"真实权重: [2.0, 3.0, -1.0]")
print(f"学习到的偏置: {model.linear.bias.item():.4f}")
print(f"真实偏置: 0.5000")
```

## 感知机：线性分类器

**感知机**（Perceptron）是最早的神经网络模型之一，用于二分类问题。

### 模型定义

$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b) = \begin{cases} +1 & \text{if } \mathbf{w}^T \mathbf{x} + b > 0 \\ -1 & \text{if } \mathbf{w}^T \mathbf{x} + b < 0 \end{cases}$$

**几何意义**：$\mathbf{w}^T \mathbf{x} + b = 0$ 定义了一个**超平面**，将空间分成两部分。

### 物理类比

感知机类似于一个**阈值检测器**：
- 输入信号 $\mathbf{x}$ 经过加权求和
- 与阈值 $-b$ 比较
- 输出二值响应

这与物理中的许多系统类似：
- 神经元的激活（积分-发放模型）
- 相变的临界行为
- 量子系统的测量（投影到本征态）

### 逻辑回归：概率化的感知机

感知机的输出是离散的（±1），这在训练时不方便（不可微）。**逻辑回归**使用 Sigmoid 函数将输出变为连续的概率值：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

其中 Sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$。

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 二分类逻辑回归
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 生成分类数据
torch.manual_seed(42)
N = 200

# 类别 0: 中心在 (-1, -1)
X0 = torch.randn(N // 2, 2) * 0.5 + torch.tensor([-1.0, -1.0])
y0 = torch.zeros(N // 2, 1)

# 类别 1: 中心在 (1, 1)
X1 = torch.randn(N // 2, 2) * 0.5 + torch.tensor([1.0, 1.0])
y1 = torch.ones(N // 2, 1)

X = torch.cat([X0, X1], dim=0)
y = torch.cat([y0, y1], dim=0)

# 创建模型
model = LogisticRegression(input_dim=2)

# 二分类交叉熵损失
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# 训练
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        accuracy = ((y_pred > 0.5) == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, accuracy = {accuracy.item():.2%}")

# 决策边界
w = model.linear.weight.data.flatten()
b = model.linear.bias.item()
print(f"\n决策边界: {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {b:.2f} = 0")
```

## 线性模型的局限性

### 异或问题 (XOR Problem)

感知机和线性分类器有一个根本的局限：**无法解决非线性可分问题**。

经典的例子是**异或问题**：

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|-------|-------|---------------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

无论如何选择 $w_1, w_2, b$，都不可能用一条直线将两类点分开。

```python
import torch
import torch.nn as nn

# XOR 数据
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# 尝试用逻辑回归解决（会失败）
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# 训练
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("线性模型预测结果:")
print(f"  (0,0) -> {model(X[0]).item():.4f} (期望: 0)")
print(f"  (0,1) -> {model(X[1]).item():.4f} (期望: 1)")
print(f"  (1,0) -> {model(X[2]).item():.4f} (期望: 1)")
print(f"  (1,1) -> {model(X[3]).item():.4f} (期望: 0)")
print("\n线性模型无法解决 XOR 问题！")
```

### 解决方案：多层网络

要解决 XOR 问题，需要引入**隐藏层**和**非线性激活函数**：

```python
# 多层感知机解决 XOR
model_mlp = nn.Sequential(
    nn.Linear(2, 4),   # 隐藏层：2 -> 4
    nn.ReLU(),          # 非线性激活
    nn.Linear(4, 1),   # 输出层：4 -> 1
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model_mlp(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\n多层感知机预测结果:")
print(f"  (0,0) -> {model_mlp(X[0]).item():.4f} (期望: 0)")
print(f"  (0,1) -> {model_mlp(X[1]).item():.4f} (期望: 1)")
print(f"  (1,0) -> {model_mlp(X[2]).item():.4f} (期望: 1)")
print(f"  (1,1) -> {model_mlp(X[3]).item():.4f} (期望: 0)")
print("\n多层感知机成功解决 XOR 问题！")
```

## 从物理视角理解线性模型

### 1. 最小二乘法与最小作用量原理

线性回归的最小二乘解可以看作**最小作用量原理**的体现：

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$$

解析解为：

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

这类似于物理中寻找使"作用量"最小的路径。

### 2. 正则化与弹性势能

**L2 正则化**（Ridge 回归）：

$$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$$

物理类比：
- 第一项 = 数据拟合（弹簧势能，将预测拉向数据）
- 第二项 = 正则化（另一组弹簧，将参数拉向原点）
- $\lambda$ = 正则化弹簧的劲度系数

```python
# L2 正则化示例
import torch
import torch.nn as nn

# 带 L2 正则化的线性回归
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)  # weight_decay = λ

# weight_decay 参数自动添加 L2 正则化
```

### 3. 决策边界与相变

在分类问题中，决策边界 $\mathbf{w}^T \mathbf{x} + b = 0$ 将空间分成两个区域，类似于物理中的**相变边界**：

- 边界两侧属于不同的"相"（类别）
- 远离边界，分类置信度高（深入某一相）
- 在边界附近，预测不确定（临界区域）

## nn.Linear 详解

`nn.Linear` 是 PyTorch 中最基础的层：

```python
# 定义
linear = nn.Linear(in_features=3, out_features=2, bias=True)

# 等价于: y = x @ W.T + b
# 其中 W 的形状是 [out_features, in_features] = [2, 3]
#      b 的形状是 [out_features] = [2]

print(f"权重形状: {linear.weight.shape}")  # [2, 3]
print(f"偏置形状: {linear.bias.shape}")    # [2]

# 前向传播
x = torch.randn(5, 3)  # 5 个样本，每个 3 维
y = linear(x)          # 输出形状 [5, 2]
print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
```

### 参数初始化

`nn.Linear` 默认使用 **Kaiming 均匀初始化**：

```python
# 查看默认初始化
linear = nn.Linear(100, 50)
print(f"权重均值: {linear.weight.mean().item():.6f}")
print(f"权重标准差: {linear.weight.std().item():.4f}")

# 自定义初始化
nn.init.xavier_normal_(linear.weight)
nn.init.zeros_(linear.bias)
print(f"Xavier 初始化后标准差: {linear.weight.std().item():.4f}")
```

## 本节小结

| 概念 | 说明 |
|------|------|
| 线性回归 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$，用于回归问题 |
| 感知机 | $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$，二分类 |
| 逻辑回归 | $P(y=1) = \sigma(\mathbf{w}^T \mathbf{x} + b)$，概率输出 |
| 决策边界 | 超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ |
| 局限性 | 无法处理非线性可分问题（如 XOR） |
| `nn.Linear` | PyTorch 中的线性层实现 |

## 练习

1. 实现一个完整的线性回归，拟合物理数据 $y = A\sin(\omega x) + B$（提示：将 $\sin(\omega x)$ 作为特征输入）
2. 用逻辑回归实现一个简单的粒子分类器：根据动量 $(p_x, p_y, p_z)$ 和能量 $E$ 判断粒子类型
3. 验证当数据线性可分时，逻辑回归的权重会趋向于无穷大（过拟合现象）

## 延伸阅读

- [PyTorch nn.Linear 文档](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

---

[返回章节目录](./README.md) | [下一节：激活函数 →](./02_activation_functions.md)

