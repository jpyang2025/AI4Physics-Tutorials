# 2.2 激活函数

## 为什么需要激活函数？

在上一节中，我们看到线性模型无法解决 XOR 这样的非线性问题。原因很简单：

**多层线性变换的复合仍然是线性变换。**

设 $f_1(\mathbf{x}) = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ 和 $f_2(\mathbf{x}) = \mathbf{W}_2 \mathbf{x} + \mathbf{b}_2$，则：

$$f_2(f_1(\mathbf{x})) = \mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2)$$

这仍然是一个线性函数！无论堆叠多少层，都等价于单层线性变换。

**激活函数**引入非线性，使得网络能够逼近任意复杂的函数。

## 物理视角：响应函数

在物理学中，许多系统的响应都是非线性的：

| 物理系统 | 响应函数 | 类似的激活函数 |
|---------|---------|--------------|
| 神经元动作电位 | 阈值激发，全或无 | 阶跃函数、Sigmoid |
| 整流器（二极管） | 单向导通 | ReLU |
| 饱和放大器 | 低信号放大，高信号饱和 | Tanh |
| 量子隧穿 | 指数衰减 | Softplus |

激活函数就像是神经网络中的"非线性元件"。

## 常用激活函数

### 1. Sigmoid 函数

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**性质**：
- 输出范围：$(0, 1)$
- 导数：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- 最大导数值：$\sigma'(0) = 0.25$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-6, 6, 100)

# Sigmoid
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)

# 导数
x_grad = torch.linspace(-6, 6, 100, requires_grad=True)
y_grad = sigmoid(x_grad)
y_grad.sum().backward()
dy_sigmoid = x_grad.grad

print(f"Sigmoid(0) = {sigmoid(torch.tensor(0.0)).item():.4f}")  # 0.5
print(f"Sigmoid(-∞) → 0, Sigmoid(+∞) → 1")
print(f"最大导数 (在 x=0): {dy_sigmoid[50].item():.4f}")  # 0.25
```

**优点**：
- 输出可解释为概率
- 平滑可微

**缺点**：
- **梯度消失**：当 $|x|$ 较大时，$\sigma'(x) \approx 0$
- 输出不以零为中心
- 计算涉及指数运算，较慢

**使用场景**：
- 二分类的输出层
- 需要概率输出时

### 2. Tanh 函数

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**性质**：
- 输出范围：$(-1, 1)$
- 导数：$\tanh'(x) = 1 - \tanh^2(x)$
- 最大导数值：$\tanh'(0) = 1$

```python
tanh = nn.Tanh()
y_tanh = tanh(x)

print(f"Tanh(0) = {tanh(torch.tensor(0.0)).item():.4f}")  # 0
print(f"Tanh(-∞) → -1, Tanh(+∞) → 1")
```

**优点**：
- 输出以零为中心（有助于下一层的学习）
- 梯度比 Sigmoid 大

**缺点**：
- 仍有梯度消失问题
- 计算较慢

**物理类比**：Tanh 类似于磁化强度随外场的变化（顺磁性材料的 Langevin 函数）。

### 3. ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**性质**：
- 输出范围：$[0, +\infty)$
- 导数：$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$（在 $x=0$ 处不可微，但实践中忽略）

```python
relu = nn.ReLU()
y_relu = relu(x)

print(f"ReLU(-2) = {relu(torch.tensor(-2.0)).item()}")  # 0
print(f"ReLU(2) = {relu(torch.tensor(2.0)).item()}")    # 2
```

**优点**：
- 计算极快（只需比较和赋值）
- 正区域梯度恒为 1，缓解梯度消失
- 稀疏激活（部分神经元输出为 0），有正则化效果

**缺点**：
- **Dead ReLU 问题**：如果神经元输入始终为负，梯度永远为 0，无法学习
- 输出不以零为中心

**物理类比**：理想二极管，只允许单向电流通过。

### 4. Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

其中 $\alpha$ 通常取 0.01。

```python
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
y_leaky = leaky_relu(x)

print(f"LeakyReLU(-2) = {leaky_relu(torch.tensor(-2.0)).item()}")  # -0.02
print(f"LeakyReLU(2) = {leaky_relu(torch.tensor(2.0)).item()}")    # 2
```

**优点**：解决 Dead ReLU 问题，负区域也有梯度。

### 5. ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

```python
elu = nn.ELU(alpha=1.0)
y_elu = elu(x)
```

**优点**：
- 负区域有非零输出，均值接近零
- 平滑可微

### 6. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

```python
gelu = nn.GELU()
y_gelu = gelu(x)
```

**优点**：
- 在 Transformer 等现代架构中表现优异
- 平滑非单调

**物理类比**：GELU 类似于高斯噪声背景下的信号门控。

### 7. Softplus

$$\text{Softplus}(x) = \ln(1 + e^x)$$

这是 ReLU 的平滑近似。

```python
softplus = nn.Softplus()
y_softplus = softplus(x)

print(f"Softplus(x) ≈ ReLU(x) 当 x >> 0")
print(f"Softplus(0) = ln(2) = {softplus(torch.tensor(0.0)).item():.4f}")
```

### 8. Swish / SiLU

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```python
swish = nn.SiLU()  # PyTorch 中称为 SiLU
y_swish = swish(x)
```

**优点**：
- 自门控机制
- 在深层网络中表现良好

## 激活函数可视化对比

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 200)

# 创建激活函数
activations = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'GELU': nn.GELU(),
    'Softplus': nn.Softplus(),
    'SiLU/Swish': nn.SiLU()
}

# 绘图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, (name, func) in zip(axes, activations.items()):
    y = func(x)
    ax.plot(x.numpy(), y.detach().numpy(), 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name, fontsize=14)
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()
```

## 激活函数的导数

```python
import torch

def compute_gradient(activation, x_val):
    """计算激活函数在某点的导数"""
    x = torch.tensor([x_val], requires_grad=True)
    y = activation(x)
    y.backward()
    return x.grad.item()

# 在 x=0 处的导数
print("激活函数在 x=0 处的导数:")
print(f"  Sigmoid: {compute_gradient(nn.Sigmoid(), 0.0):.4f}")  # 0.25
print(f"  Tanh:    {compute_gradient(nn.Tanh(), 0.0):.4f}")     # 1.0
print(f"  ReLU:    {compute_gradient(nn.ReLU(), 0.01):.4f}")    # 1.0 (x>0)
print(f"  GELU:    {compute_gradient(nn.GELU(), 0.0):.4f}")     # 0.5
print(f"  Swish:   {compute_gradient(nn.SiLU(), 0.0):.4f}")     # 0.5
```

## 梯度消失问题

**梯度消失**是深层网络训练的主要障碍之一。当使用 Sigmoid 或 Tanh 时：

```python
import torch
import torch.nn as nn

# 演示梯度消失
def demonstrate_vanishing_gradient(activation, n_layers=10):
    """演示多层网络中的梯度消失"""
    x = torch.tensor([1.0], requires_grad=True)
    
    for _ in range(n_layers):
        x = activation(x)
    
    x.backward()
    return x.grad.item()

print("10层网络中输入的梯度:")
print(f"  Sigmoid: {demonstrate_vanishing_gradient(nn.Sigmoid(), 10):.2e}")
print(f"  Tanh:    {demonstrate_vanishing_gradient(nn.Tanh(), 10):.2e}")
print(f"  ReLU:    {demonstrate_vanishing_gradient(nn.ReLU(), 10):.2e}")

# Sigmoid 的梯度会指数级衰减！
# 10 层后，梯度约为 0.25^10 ≈ 10^-6
```

**为什么 ReLU 能缓解梯度消失？**

- Sigmoid/Tanh 的导数最大值 < 1，多层相乘会趋近于 0
- ReLU 正区域导数 = 1，梯度可以无损传递

## 如何选择激活函数？

### 实践指南

| 场景 | 推荐激活函数 |
|------|------------|
| 隐藏层（默认选择） | ReLU 或其变体（LeakyReLU, ELU） |
| 深层网络隐藏层 | GELU, Swish |
| Transformer | GELU |
| 二分类输出层 | Sigmoid |
| 多分类输出层 | Softmax |
| 回归输出层 | 无激活函数（恒等映射） |
| RNN/LSTM 门控 | Sigmoid, Tanh |
| 需要概率输出 | Sigmoid (0-1), Softmax (多类) |

### 决策流程

```
开始
  ├── 输出层？
  │     ├── 是 → 二分类？ → Sigmoid
  │     │         多分类？ → Softmax
  │     │         回归？   → 无激活
  │     │
  │     └── 否（隐藏层）
  │           ├── 普通 CNN/MLP → ReLU / LeakyReLU
  │           ├── Transformer → GELU
  │           ├── RNN 门控 → Sigmoid/Tanh
  │           └── 生成模型 → LeakyReLU / Tanh
```

## 输出层激活函数

### Softmax：多分类

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

将任意实数向量转换为概率分布。

```python
# Softmax 示例
logits = torch.tensor([2.0, 1.0, 0.1])
softmax = nn.Softmax(dim=0)
probs = softmax(logits)

print(f"Logits: {logits.tolist()}")
print(f"Probabilities: {probs.tolist()}")
print(f"Sum: {probs.sum().item()}")  # 1.0

# 物理类比：Boltzmann 分布
# P(state_i) ∝ exp(-E_i / kT)
# 这里 logits 类似于 -E/kT
```

**物理类比**：Softmax 与统计力学中的 **Boltzmann 分布**完全等价：

$$P_i = \frac{e^{-E_i/k_BT}}{\sum_j e^{-E_j/k_BT}}$$

如果将 $z_i = -E_i/k_BT$，则 Softmax 就是 Boltzmann 分布！

### 温度参数

```python
def softmax_with_temperature(logits, temperature=1.0):
    """带温度的 Softmax"""
    return nn.functional.softmax(logits / temperature, dim=0)

logits = torch.tensor([2.0, 1.0, 0.0])

print("不同温度下的概率分布:")
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"  T={T:4.1f}: {probs.tolist()}")

# T → 0: 确定性（只选最大值）
# T → ∞: 均匀分布
# 类似于热力学中的温度效应！
```

## 使用激活函数

### 函数式 API

```python
import torch.nn.functional as F

x = torch.randn(5, 10)

# 函数式调用
y1 = F.relu(x)
y2 = F.sigmoid(x)
y3 = F.gelu(x)
y4 = F.softmax(x, dim=1)  # 需要指定维度
```

### 模块式 API

```python
# 作为模块使用
relu = nn.ReLU()
y = relu(x)

# 在 Sequential 中使用
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.GELU(),
    nn.Linear(20, 5),
    nn.Softmax(dim=1)
)
```

### 原地操作

```python
# inplace=True 可以节省内存，但要小心使用
relu_inplace = nn.ReLU(inplace=True)

x = torch.randn(5, 10)
y = relu_inplace(x)  # x 会被修改！

# 注意：如果 x 需要梯度，inplace 可能导致问题
```

## 本节小结

| 激活函数 | 公式 | 输出范围 | 主要优点 | 主要缺点 |
|---------|------|---------|---------|---------|
| Sigmoid | $1/(1+e^{-x})$ | $(0,1)$ | 概率解释 | 梯度消失 |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $(-1,1)$ | 零中心 | 梯度消失 |
| ReLU | $\max(0,x)$ | $[0,\infty)$ | 计算快，缓解梯度消失 | Dead ReLU |
| LeakyReLU | $\max(\alpha x, x)$ | $(-\infty,\infty)$ | 解决 Dead ReLU | 需调参 |
| GELU | $x\Phi(x)$ | $(-0.17,\infty)$ | 平滑，现代架构常用 | 计算较慢 |
| Softmax | $e^{z_i}/\sum e^{z_j}$ | $(0,1)$，和为1 | 多分类概率 | 仅用于输出层 |

## 练习

1. 绘制 ReLU、LeakyReLU、ELU 的导数图，比较它们在 $x<0$ 区域的行为
2. 实现一个自定义激活函数 $f(x) = x \cdot \tanh(x)$（称为 Mish），并验证其平滑性
3. 用实验验证：使用 Sigmoid 激活的 20 层网络，输入层的梯度有多小？

## 延伸阅读

- Nair & Hinton (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines"
- Hendrycks & Gimpel (2016). "Gaussian Error Linear Units (GELUs)"
- Ramachandran et al. (2017). "Searching for Activation Functions"

---

[← 上一节：感知机与线性模型](./01_perceptron.md) | [下一节：前馈神经网络 →](./03_feedforward_networks.md)

