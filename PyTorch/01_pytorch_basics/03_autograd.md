# 1.3 自动微分 (Autograd)

## 为什么需要自动微分？

神经网络的训练本质上是一个**优化问题**：找到一组参数 $\theta$，使得损失函数 $L(\theta)$ 最小化。最常用的优化方法是**梯度下降**：

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta L$$

其中 $\eta$ 是学习率，$\nabla_\theta L$ 是损失函数对参数的梯度。

### 物理视角

想象一个在势能面上运动的粒子：
- **损失函数** $L(\theta)$ = 势能 $U(\mathbf{r})$
- **参数** $\theta$ = 粒子位置 $\mathbf{r}$
- **梯度** $\nabla L$ = 势能梯度，指向势能增加最快的方向
- **负梯度** $-\nabla L$ = 力的方向（指向势能下降最快的方向）
- **梯度下降** = 粒子在阻尼作用下沿势能面滚向最低点

PyTorch 的 **Autograd** 系统可以自动计算这些梯度，无需手动推导和编程。

## 计算图与反向传播

### 计算图 (Computational Graph)

PyTorch 在执行运算时会构建一个**动态计算图**，记录所有操作：

```python
import torch

# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 前向传播：执行计算
z = x * y      # z = x * y
w = z + x      # w = z + x = x*y + x
v = w ** 2     # v = w^2 = (x*y + x)^2

# 此时，PyTorch 已经记录了整个计算过程
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"z = x*y = {z.item()}")
print(f"w = z+x = {w.item()}")
print(f"v = w^2 = {v.item()}")
```

### 反向传播 (Backward Pass)

调用 `.backward()` 计算梯度：

```python
# 反向传播：计算梯度
v.backward()

# v = (x*y + x)^2
# ∂v/∂x = 2(x*y + x)(y + 1) = 2*8*4 = 64
# ∂v/∂y = 2(x*y + x)*x = 2*8*2 = 32

print(f"∂v/∂x = {x.grad.item()}")  # 64.0
print(f"∂v/∂y = {y.grad.item()}")  # 32.0
```

### 数学验证

让我们手动验证一下：

$$v = (xy + x)^2 = x^2(y+1)^2$$

$$\frac{\partial v}{\partial x} = 2x(y+1)^2 = 2 \times 2 \times 16 = 64 \quad \checkmark$$

$$\frac{\partial v}{\partial y} = 2x^2(y+1) = 2 \times 4 \times 4 = 32 \quad \checkmark$$

## requires_grad：追踪梯度

`requires_grad=True` 告诉 PyTorch 追踪这个张量上的所有操作：

```python
# 创建张量，指定需要梯度
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"requires_grad: {a.requires_grad}")

# 检查张量是否是叶子节点
print(f"is_leaf: {a.is_leaf}")

# 由其他张量计算得到的张量
b = a * 2
print(f"b.requires_grad: {b.requires_grad}")  # True（继承）
print(f"b.is_leaf: {b.is_leaf}")  # False（不是叶子）

# 查看梯度函数
print(f"b.grad_fn: {b.grad_fn}")  # <MulBackward0>
```

### 叶子节点与非叶子节点

```python
x = torch.tensor([1.0], requires_grad=True)  # 叶子节点
y = x * 2  # 非叶子节点
z = y * 3  # 非叶子节点

z.backward()

# 只有叶子节点默认保存梯度
print(f"x.grad: {x.grad}")  # tensor([6.])
print(f"y.grad: {y.grad}")  # None（非叶子节点默认不保存）

# 如果需要中间节点的梯度，使用 retain_grad()
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.retain_grad()  # 告诉 PyTorch 保留 y 的梯度
z = y * 3
z.backward()

print(f"y.grad (with retain_grad): {y.grad}")  # tensor([3.])
```

## 梯度计算详解

### 标量输出的梯度

当输出是标量时，可以直接调用 `backward()`：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # 标量输出

y.backward()
print(f"∂y/∂x = {x.grad}")  # [2., 4., 6.]
# y = x1^2 + x2^2 + x3^2
# ∂y/∂xi = 2*xi
```

### 非标量输出的梯度

当输出不是标量时，需要提供 `gradient` 参数（雅可比矩阵的一行）：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # 向量输出 [1, 4, 9]

# 直接 backward() 会报错：
# y.backward()  # RuntimeError!

# 需要提供 gradient 参数
# 这实际上计算的是 v^T @ (∂y/∂x)，其中 v 是 gradient
gradient = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient)
print(f"梯度: {x.grad}")  # [2., 4., 6.]

# 这等价于：
# ∂y/∂x 是对角矩阵 diag([2x1, 2x2, 2x3])
# gradient @ (∂y/∂x) = [1,1,1] @ diag([2,4,6]) = [2,4,6]
```

### 多次反向传播与梯度累积

**重要**：梯度是累积的！每次调用 `backward()` 会将梯度**加到**现有梯度上：

```python
x = torch.tensor([1.0], requires_grad=True)

# 第一次
y1 = x * 2
y1.backward()
print(f"第一次 backward 后: {x.grad}")  # tensor([2.])

# 第二次 - 梯度累积！
y2 = x * 3
y2.backward()
print(f"第二次 backward 后: {x.grad}")  # tensor([5.]) = 2 + 3

# 清零梯度
x.grad.zero_()
print(f"清零后: {x.grad}")  # tensor([0.])

# 第三次
y3 = x * 4
y3.backward()
print(f"第三次 backward 后: {x.grad}")  # tensor([4.])
```

**为什么设计成累积？** 这在处理小批量（mini-batch）时很有用——可以累积多个样本的梯度后再更新参数。

## 停止梯度追踪

有时我们不需要计算梯度（如模型推理），可以通过以下方式：

### torch.no_grad()

```python
x = torch.tensor([1.0], requires_grad=True)

# 在 no_grad 上下文中，不追踪梯度
with torch.no_grad():
    y = x * 2
    print(f"y.requires_grad: {y.requires_grad}")  # False

# 这在模型推理时很重要：
# - 节省内存（不存储中间结果）
# - 加速计算
```

### detach()

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2

# detach() 创建一个不需要梯度的副本
y_detached = y.detach()
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False

# 注意：y_detached 与 y 共享数据！
y_detached[0] = 100
print(f"y after modifying y_detached: {y}")  # tensor([100.])
```

### 直接设置 requires_grad

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(f"初始 requires_grad: {x.requires_grad}")  # False

# 原地修改
x.requires_grad_(True)
print(f"修改后 requires_grad: {x.requires_grad}")  # True
```

## 物理应用：梯度下降找极值

让我们用自动微分解一个优化问题——找函数的最小值：

```python
import torch
import matplotlib.pyplot as plt

# 定义势能函数 V(x) = x^4 - 10*x^2 + 5（双势阱）
def potential(x):
    return x**4 - 10*x**2 + 5

# 初始位置
x = torch.tensor([3.0], requires_grad=True)

# 梯度下降参数
learning_rate = 0.01
n_steps = 200

# 记录轨迹
trajectory = [x.item()]
energies = [potential(x).item()]

# 梯度下降循环
for step in range(n_steps):
    # 1. 计算势能（前向传播）
    V = potential(x)
    
    # 2. 计算梯度（反向传播）
    V.backward()
    
    # 3. 更新位置（不追踪这个操作的梯度）
    with torch.no_grad():
        x -= learning_rate * x.grad  # x_new = x - η * ∂V/∂x
    
    # 4. 清零梯度（为下一次迭代准备）
    x.grad.zero_()
    
    # 记录
    trajectory.append(x.item())
    energies.append(potential(x).item())

print(f"初始位置: {trajectory[0]:.4f}")
print(f"最终位置: {trajectory[-1]:.4f}")
print(f"初始势能: {energies[0]:.4f}")
print(f"最终势能: {energies[-1]:.4f}")

# 解析解：dV/dx = 4x^3 - 20x = 0 => x = 0, ±√5 ≈ ±2.236
print(f"理论极值点: ±{5**0.5:.4f}")
```

## 高阶导数

PyTorch 支持计算高阶导数：

```python
x = torch.tensor([2.0], requires_grad=True)

# 一阶导数
y = x ** 3  # y = x^3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 3x^2 = {dy_dx.item()}")  # 12.0

# 二阶导数
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
print(f"d²y/dx² = 6x = {d2y_dx2.item()}")  # 12.0

# 三阶导数
d3y_dx3 = torch.autograd.grad(d2y_dx2, x)[0]
print(f"d³y/dx³ = 6 = {d3y_dx3.item()}")  # 6.0
```

### 物理应用：计算力和力常数

```python
# 简谐振子势能 V(x) = 1/2 * k * x^2
k = 2.0  # 弹簧常数
x = torch.tensor([3.0], requires_grad=True)

V = 0.5 * k * x ** 2

# 力 F = -dV/dx
grad_V = torch.autograd.grad(V, x, create_graph=True)[0]
F = -grad_V
print(f"力 F = -kx = {F.item()}")  # -6.0

# 力常数（刚度）= -dF/dx = d²V/dx²
d2V_dx2 = torch.autograd.grad(grad_V, x)[0]
print(f"力常数 = {d2V_dx2.item()}")  # 2.0 = k ✓
```

## 雅可比矩阵与向量积

对于向量函数 $\mathbf{y} = f(\mathbf{x})$，完整的雅可比矩阵可能非常大。PyTorch 提供了高效计算**雅可比矩阵-向量积 (JVP)** 和**向量-雅可比矩阵积 (VJP)** 的方法：

```python
from torch.autograd.functional import jacobian, vjp, jvp

# 定义向量函数
def f(x):
    return torch.stack([
        x[0] * x[1],      # y1 = x1 * x2
        x[0] + x[1]**2    # y2 = x1 + x2^2
    ])

x = torch.tensor([2.0, 3.0])

# 完整的雅可比矩阵
J = jacobian(f, x)
print(f"雅可比矩阵:\n{J}")
# [[∂y1/∂x1, ∂y1/∂x2],   [[x2, x1],      [[3, 2],
#  [∂y2/∂x1, ∂y2/∂x2]] =  [1, 2*x2]]  =   [1, 6]]

# VJP: v^T @ J
v = torch.tensor([1.0, 1.0])
output, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(v)
print(f"VJP (v^T @ J): {vjp_result}")

# JVP: J @ u  
u = torch.tensor([1.0, 0.0])
output, jvp_result = jvp(f, (x,), (u,))
print(f"JVP (J @ u): {jvp_result}")
```

## 自定义自动微分函数

有时需要定义自己的前向和反向传播逻辑：

```python
class MySigmoid(torch.autograd.Function):
    """
    自定义 Sigmoid 函数
    σ(x) = 1 / (1 + e^(-x))
    σ'(x) = σ(x) * (1 - σ(x))
    """
    
    @staticmethod
    def forward(ctx, x):
        # ctx 用于保存反向传播需要的值
        sigmoid = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(sigmoid)
        return sigmoid
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是损失对输出的梯度
        sigmoid, = ctx.saved_tensors
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        return grad_input

# 使用自定义函数
x = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
y = MySigmoid.apply(x)
loss = y.sum()
loss.backward()

print(f"输入: {x.data}")
print(f"输出: {y.data}")
print(f"梯度: {x.grad}")

# 验证与 PyTorch 内置 sigmoid 一致
x2 = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
y2 = torch.sigmoid(x2)
y2.sum().backward()
print(f"PyTorch sigmoid 梯度: {x2.grad}")
```

## 常见陷阱与最佳实践

### 1. 忘记清零梯度

```python
# 错误示例
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x * 2
    y.backward()
    print(f"第 {i+1} 次: x.grad = {x.grad}")  # 2, 4, 6（累积！）

# 正确做法
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    if x.grad is not None:
        x.grad.zero_()  # 清零梯度
    y = x * 2
    y.backward()
    print(f"第 {i+1} 次: x.grad = {x.grad}")  # 2, 2, 2
```

### 2. 原地操作破坏计算图

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2

# 这会报错或产生错误结果
# y.add_(1)  # 原地操作破坏了计算图

# 正确做法
y = y + 1  # 创建新张量
```

### 3. 在更新参数时追踪梯度

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()

# 错误：更新时会被追踪
# x = x - 0.1 * x.grad  # 这会创建新的计算图！

# 正确做法
with torch.no_grad():
    x -= 0.1 * x.grad  # 或 x.data -= 0.1 * x.grad
```

### 4. 混合 NumPy 操作

```python
x = torch.tensor([1.0], requires_grad=True)

# 错误：NumPy 操作会断开计算图
# y = np.sin(x.numpy())  # x.numpy() 需要 detach

# 正确做法：使用 PyTorch 的函数
y = torch.sin(x)
```

## 本节小结

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 告诉 PyTorch 追踪此张量上的操作 |
| `.backward()` | 反向传播，计算梯度 |
| `.grad` | 存储计算出的梯度 |
| `.grad.zero_()` | 清零梯度（每次更新前必须做） |
| `torch.no_grad()` | 上下文管理器，禁止梯度追踪 |
| `.detach()` | 创建不需要梯度的副本 |
| `create_graph=True` | 允许计算高阶导数 |

## 练习

1. 使用梯度下降找到函数 $f(x) = (x-3)^2 + 2$ 的最小值点
2. 计算函数 $f(x, y) = x^2 y + xy^2$ 在点 $(1, 2)$ 处的梯度
3. 实现自定义的 ReLU 函数（`max(0, x)`），包括前向和反向传播

## 延伸阅读

- [PyTorch Autograd 教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Autograd 机制详解](https://pytorch.org/docs/stable/notes/autograd.html)
- [自定义 Function 文档](https://pytorch.org/docs/stable/autograd.html#function)

---

[← 上一节：张量运算](./02_tensor_operations.md) | [返回章节目录](./README.md)

