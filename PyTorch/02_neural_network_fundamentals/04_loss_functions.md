# 2.4 损失函数

## 什么是损失函数？

**损失函数**（Loss Function），也称为**代价函数**（Cost Function）或**目标函数**（Objective Function），用于量化模型预测值与真实值之间的差异。

神经网络训练的目标就是**最小化损失函数**：

$$\theta^* = \arg\min_\theta L(\theta)$$

## 物理视角：能量最小化

损失函数与物理中的**能量**概念高度类似：

| 物理概念 | 机器学习概念 |
|---------|-------------|
| 能量 $E$ | 损失 $L$ |
| 系统构型 $\mathbf{r}$ | 模型参数 $\theta$ |
| 力 $\mathbf{F} = -\nabla E$ | 负梯度 $-\nabla_\theta L$ |
| 势能最小化 | 损失最小化 |
| 基态 | 全局最优解 |
| 亚稳态 | 局部极小值 |

**训练过程就像在能量面上寻找最低点的过程。**

## 回归问题的损失函数

### 1. 均方误差 (MSE)

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

```python
import torch
import torch.nn as nn

# 预测值和真实值
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
y_true = torch.tensor([3.0, -0.5, 2.0, 7.5])

# 方法1：手动计算
mse_manual = ((y_pred - y_true) ** 2).mean()
print(f"手动计算 MSE: {mse_manual.item():.4f}")

# 方法2：使用 PyTorch
criterion = nn.MSELoss()
mse = criterion(y_pred, y_true)
print(f"nn.MSELoss: {mse.item():.4f}")

# 方法3：使用函数式 API
import torch.nn.functional as F
mse_f = F.mse_loss(y_pred, y_true)
print(f"F.mse_loss: {mse_f.item():.4f}")
```

**物理类比**：弹簧势能

$$E = \frac{1}{2}k(x - x_0)^2$$

每个数据点就像一个弹簧，将预测值"拉向"真实值。MSE 对大误差惩罚更重（平方关系）。

**优点**：
- 数学性质好，处处可微
- 计算简单

**缺点**：
- 对异常值敏感（大误差被平方放大）

### 2. 平均绝对误差 (MAE / L1 Loss)

$$L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

```python
# MAE
mae = F.l1_loss(y_pred, y_true)
print(f"MAE: {mae.item():.4f}")

# 或使用类
criterion_l1 = nn.L1Loss()
mae_nn = criterion_l1(y_pred, y_true)
```

**物理类比**：V 形势阱

**优点**：
- 对异常值更鲁棒
- 误差线性增长

**缺点**：
- 在零点不可微（但实践中不是大问题）
- 梯度恒定，可能导致训练末期震荡

### 3. Huber Loss（平滑 L1 Loss）

结合 MSE 和 MAE 的优点：

$$L_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & |y - \hat{y}| > \delta \end{cases}$$

```python
# Huber Loss
huber = F.huber_loss(y_pred, y_true, delta=1.0)
print(f"Huber Loss: {huber.item():.4f}")

# 或
criterion_huber = nn.HuberLoss(delta=1.0)
```

**物理类比**：近似谐振子势，远处变成线性

**优点**：
- 小误差区域像 MSE（平滑梯度）
- 大误差区域像 MAE（鲁棒性）

### 4. 对数余弦损失 (Log-Cosh)

$$L_{\text{log-cosh}} = \sum_{i=1}^{N} \log(\cosh(y_i - \hat{y}_i))$$

```python
def log_cosh_loss(y_pred, y_true):
    return torch.log(torch.cosh(y_pred - y_true)).mean()

log_cosh = log_cosh_loss(y_pred, y_true)
print(f"Log-Cosh Loss: {log_cosh.item():.4f}")
```

**性质**：类似 Huber，但处处二阶可微

## 分类问题的损失函数

### 1. 二分类交叉熵 (BCE)

对于二分类问题，设 $p$ 为预测为正类的概率：

$$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

```python
# 预测概率（经过 Sigmoid）
p = torch.tensor([0.9, 0.2, 0.7, 0.1])
y = torch.tensor([1.0, 0.0, 1.0, 0.0])

# BCE Loss
criterion_bce = nn.BCELoss()
bce = criterion_bce(p, y)
print(f"BCE Loss: {bce.item():.4f}")

# 更稳定的版本：接受 logits（未经 Sigmoid）
logits = torch.tensor([2.0, -1.5, 1.0, -2.0])
criterion_bce_logits = nn.BCEWithLogitsLoss()
bce_logits = criterion_bce_logits(logits, y)
print(f"BCE with Logits: {bce_logits.item():.4f}")
```

**物理类比**：信息熵

从信息论的角度，交叉熵测量的是用预测分布编码真实分布所需的"信息量"。

$$H(p, q) = -\sum_x p(x) \log q(x)$$

当预测完美时（$q = p$），交叉熵等于真实分布的熵。

### 2. 多分类交叉熵 (Cross-Entropy)

对于 $C$ 类分类问题：

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$$

其中 $y_{i,c}$ 是 one-hot 编码。实际上简化为：

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i, y_i})$$

```python
# 多分类
logits = torch.tensor([
    [2.0, 1.0, 0.1],   # 样本1
    [0.5, 2.5, 0.3],   # 样本2
    [0.2, 0.3, 2.0]    # 样本3
])
labels = torch.tensor([0, 1, 2])  # 真实类别

# Cross-Entropy Loss（内部包含 Softmax）
criterion_ce = nn.CrossEntropyLoss()
ce_loss = criterion_ce(logits, labels)
print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

# 验证：手动计算
probs = F.softmax(logits, dim=1)
manual_ce = -torch.log(probs[range(3), labels]).mean()
print(f"手动计算: {manual_ce.item():.4f}")
```

**重要**：`nn.CrossEntropyLoss` 期望输入是**未归一化的 logits**，内部会自动应用 Softmax！

### 3. 负对数似然 (NLL Loss)

如果你已经手动计算了 log-probabilities：

```python
# 手动计算 log-softmax
log_probs = F.log_softmax(logits, dim=1)

# NLL Loss
criterion_nll = nn.NLLLoss()
nll_loss = criterion_nll(log_probs, labels)
print(f"NLL Loss: {nll_loss.item():.4f}")

# CrossEntropyLoss = LogSoftmax + NLLLoss
```

### 物理视角：极大似然与自由能

交叉熵损失等价于**最大化对数似然**：

$$L = -\log P(\text{data} | \text{model})$$

从统计力学的角度：
- **负对数似然** $\leftrightarrow$ **自由能**
- 最小化交叉熵 $\leftrightarrow$ 最小化自由能

玻尔兹曼分布：$P_i \propto e^{-E_i/k_BT}$

Softmax 分布：$P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

如果令 $z_i = -E_i/k_BT$，两者等价！

## 特殊损失函数

### 1. KL 散度 (Kullback-Leibler Divergence)

$$D_{\text{KL}}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

```python
# KL 散度
p = torch.tensor([0.4, 0.3, 0.3])
q = torch.tensor([0.33, 0.33, 0.34])

kl_div = F.kl_div(q.log(), p, reduction='sum')
print(f"KL Divergence: {kl_div.item():.4f}")

# 注意：PyTorch 的 kl_div 期望输入是 log(Q)
```

**物理意义**：KL 散度测量两个概率分布的"距离"（但不是对称的）。

### 2. 余弦相似度损失

$$L_{\text{cosine}} = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```python
# 余弦相似度损失
criterion_cosine = nn.CosineEmbeddingLoss()

a = torch.randn(5, 128)
b = torch.randn(5, 128)
target = torch.ones(5)  # 1表示相似，-1表示不相似

loss_cosine = criterion_cosine(a, b, target)
print(f"Cosine Embedding Loss: {loss_cosine.item():.4f}")
```

### 3. 对比损失 (Contrastive Loss)

用于学习相似性表示：

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, x1, x2, y):
        """
        y=1: 相似对，距离应该小
        y=0: 不相似对，距离应该大于 margin
        """
        dist = F.pairwise_distance(x1, x2)
        loss = y * dist.pow(2) + (1 - y) * F.relu(self.margin - dist).pow(2)
        return loss.mean()
```

### 4. Triplet Loss

$$L = \max(d(a, p) - d(a, n) + m, 0)$$

其中 $a$ 是锚点，$p$ 是正样本，$n$ 是负样本。

```python
criterion_triplet = nn.TripletMarginLoss(margin=1.0)

anchor = torch.randn(8, 128)
positive = torch.randn(8, 128)
negative = torch.randn(8, 128)

loss_triplet = criterion_triplet(anchor, positive, negative)
print(f"Triplet Loss: {loss_triplet.item():.4f}")
```

## 物理应用：自定义损失函数

### 1. 守恒律约束

在物理问题中，我们可能希望网络输出满足某些守恒律：

```python
class PhysicsInformedLoss(nn.Module):
    """带物理约束的损失函数"""
    
    def __init__(self, lambda_physics=1.0):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, physics_residual):
        """
        pred: 模型预测
        target: 真实值
        physics_residual: 物理方程残差（应该为0）
        """
        # 数据拟合损失
        data_loss = self.mse(pred, target)
        
        # 物理约束损失
        physics_loss = (physics_residual ** 2).mean()
        
        return data_loss + self.lambda_physics * physics_loss
```

### 2. 能量守恒约束

```python
def energy_conservation_loss(positions, velocities, masses, E_target):
    """
    能量守恒损失
    positions: [N, 3] 粒子位置
    velocities: [N, 3] 粒子速度
    masses: [N] 粒子质量
    E_target: 目标总能量
    """
    # 动能
    KE = 0.5 * (masses.unsqueeze(1) * velocities ** 2).sum()
    
    # 势能（简单的双体势）
    # ... 计算势能 PE
    
    # 总能量偏差
    E_total = KE  # + PE
    energy_violation = (E_total - E_target) ** 2
    
    return energy_violation
```

### 3. 边界条件损失

```python
def boundary_condition_loss(model, boundary_points, boundary_values):
    """
    边界条件损失
    用于 PDE 求解（如 PINN）
    """
    pred_boundary = model(boundary_points)
    return F.mse_loss(pred_boundary, boundary_values)
```

## 损失函数的选择指南

### 按任务选择

| 任务类型 | 推荐损失函数 |
|---------|-------------|
| 回归 | MSE, MAE, Huber |
| 二分类 | BCEWithLogitsLoss |
| 多分类 | CrossEntropyLoss |
| 多标签分类 | BCEWithLogitsLoss |
| 序列生成 | CrossEntropyLoss |
| 相似性学习 | ContrastiveLoss, TripletLoss |
| 概率分布匹配 | KLDivLoss |
| 物理约束问题 | MSE + 物理残差 |

### 按数据特点选择

| 数据特点 | 推荐选择 |
|---------|---------|
| 有异常值 | MAE, Huber |
| 类别不平衡 | 加权 CrossEntropy |
| 需要概率输出 | BCE, CE + Softmax |
| 需要不确定性估计 | 负对数似然 |

## reduction 参数

PyTorch 损失函数的 `reduction` 参数：

```python
# 'mean': 返回平均值（默认）
# 'sum': 返回总和
# 'none': 返回每个样本的损失

criterion = nn.MSELoss(reduction='none')
per_sample_loss = criterion(y_pred, y_true)
print(f"每个样本的损失: {per_sample_loss}")

# 'none' 在需要样本加权时很有用
weights = torch.tensor([1.0, 2.0, 1.0, 3.0])
weighted_loss = (per_sample_loss * weights).mean()
```

## 数值稳定性

### Log-Sum-Exp 技巧

计算 $\log(\sum_i e^{x_i})$ 时可能溢出：

```python
# 不稳定
x = torch.tensor([1000.0, 1001.0, 1002.0])
# naive_result = torch.log(torch.exp(x).sum())  # 会溢出！

# 稳定版本
x_max = x.max()
stable_result = x_max + torch.log(torch.exp(x - x_max).sum())
print(f"稳定结果: {stable_result.item():.4f}")

# PyTorch 内置
result = torch.logsumexp(x, dim=0)
print(f"torch.logsumexp: {result.item():.4f}")
```

### 使用 LogSoftmax + NLLLoss

比 Softmax + log + NLLLoss 更稳定：

```python
# 推荐
criterion = nn.CrossEntropyLoss()  # 内部使用 LogSoftmax

# 或
log_probs = F.log_softmax(logits, dim=1)  # 内部有稳定化处理
```

## 本节小结

| 损失函数 | 公式 | 适用场景 |
|---------|------|---------|
| MSE | $\frac{1}{N}\sum(y-\hat{y})^2$ | 回归，无异常值 |
| MAE | $\frac{1}{N}\sum\|y-\hat{y}\|$ | 回归，有异常值 |
| Huber | MSE/MAE 结合 | 回归，需要平衡 |
| BCE | $-[y\log p + (1-y)\log(1-p)]$ | 二分类 |
| CrossEntropy | $-\log p_{y}$ | 多分类 |
| KL Divergence | $\sum p \log(p/q)$ | 分布匹配 |

## 练习

1. 实现一个加权 MSE 损失，对某些样本给予更高权重
2. 对比 MSE 和 MAE 在包含异常值数据上的训练效果
3. 实现一个物理信息损失函数，约束输出满足 $\nabla \cdot \mathbf{v} = 0$（不可压缩流体）

## 延伸阅读

- [PyTorch Loss Functions 文档](https://pytorch.org/docs/stable/nn.html#loss-functions)
- Murphy, K. (2012). "Machine Learning: A Probabilistic Perspective" - Chapter 6

---

[← 上一节：前馈神经网络](./03_feedforward_networks.md) | [返回章节目录](./README.md)

