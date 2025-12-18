# 5.2 优化器

## 📖 概述

优化器是训练神经网络的核心组件，负责根据梯度更新模型参数。PyTorch 在 `torch.optim` 模块中提供了多种优化算法。

## 🎯 学习目标

- 理解优化器的工作原理
- 掌握常用优化器（SGD、Adam 等）
- 理解超参数的物理意义
- 根据问题选择合适的优化器

---

## 5.2.1 优化器基础

### 基本用法

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = nn.Linear(10, 1)

# 创建优化器，将模型参数传入
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练步骤
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# 前向传播
output = model(x)
loss = nn.MSELoss()(output, y)

# 反向传播
optimizer.zero_grad()  # 清空梯度（重要！）
loss.backward()        # 计算梯度

# 更新参数
optimizer.step()
```

### 优化器的物理直觉

优化过程可以理解为在损失函数"势能面"上的粒子运动：

$$\frac{d\theta}{dt} = -\eta \nabla L(\theta)$$

这是一个**一阶常微分方程**，描述了参数 $\theta$ 如何随"时间"（迭代步数）演化。

不同优化器对应不同的动力学系统：

| 优化器 | 动力学方程 | 物理系统 |
|--------|-----------|---------|
| SGD | $\dot{\theta} = -\eta \nabla L$ | 过阻尼运动 |
| SGD + Momentum | $m\ddot{\theta} + \gamma\dot{\theta} = -\nabla L$ | 欠阻尼振子 |
| Adam | 自适应步长 | 变质量粒子 |

---

## 5.2.2 随机梯度下降（SGD）

### 基本 SGD

```python
# 基本 SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01           # 学习率
)
```

更新规则：
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### 带动量的 SGD

动量帮助加速收敛并跳出局部极小值。

```python
# 带动量的 SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9      # 动量系数
)
```

更新规则：
$$v_t = \mu v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

**物理直觉**：动量项相当于粒子的惯性。一个滚动的球不会立即停止，而是会继续沿原方向移动一段距离。

### Nesterov 加速梯度

```python
# Nesterov 动量
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True     # 使用 Nesterov 加速
)
```

**物理直觉**：Nesterov 方法相当于先"预测"粒子下一步的位置，然后在预测位置计算梯度。这类似于辛积分器（Symplectic Integrator）的思想。

### 权重衰减（L2 正则化）

```python
# 带权重衰减的 SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2 正则化系数
)
```

等效于在损失函数中添加：
$$L_{\text{reg}} = L + \frac{\lambda}{2}\|\theta\|^2$$

**物理直觉**：权重衰减相当于给参数添加一个谐振子势能，倾向于使参数保持在原点附近。

---

## 5.2.3 Adam 优化器

Adam（Adaptive Moment Estimation）是目前最流行的优化器之一。

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # 学习率（通常比 SGD 小）
    betas=(0.9, 0.999), # 动量系数
    eps=1e-8,           # 数值稳定性
    weight_decay=0      # 权重衰减
)
```

### Adam 更新规则

```python
# Adam 算法伪代码
m = beta1 * m + (1 - beta1) * gradient     # 一阶矩估计（动量）
v = beta2 * v + (1 - beta2) * gradient**2  # 二阶矩估计（自适应学习率）

# 偏差校正
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)

# 参数更新
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

### 物理直觉

Adam 可以理解为一个**自适应步长的积分器**：

- **一阶矩 m**：累积的动量，类似于粒子的速度
- **二阶矩 v**：梯度的方差估计，用于自适应调整步长
- 在梯度变化剧烈的方向减小步长，在平坦方向增大步长

这类似于变步长数值积分方法（如 Runge-Kutta-Fehlberg），根据局部误差估计自动调整步长。

---

## 5.2.4 其他常用优化器

### AdamW

解耦权重衰减，更好的正则化效果。

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01    # 解耦的权重衰减
)
```

### RMSprop

Adam 的前身，在 RNN 中常用。

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,          # 平滑系数
    momentum=0.9
)
```

### Adagrad

自适应学习率，适合稀疏数据。

```python
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01
)
```

### LBFGS

拟牛顿法，使用二阶信息。

```python
optimizer = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100
)

# LBFGS 需要闭包函数
def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    return loss

optimizer.step(closure)
```

**物理直觉**：LBFGS 相当于使用 Hessian 矩阵信息来估计势能面的曲率，可以更快地找到极小值。

---

## 5.2.5 优化器比较

### 收敛特性

| 优化器 | 收敛速度 | 调参难度 | 适用场景 |
|--------|---------|---------|---------|
| SGD | 慢 | 较难 | 计算机视觉、追求最佳性能 |
| SGD + Momentum | 中 | 中等 | 通用 |
| Adam | 快 | 简单 | 快速原型、NLP、GAN |
| AdamW | 快 | 简单 | 需要正则化的场景 |
| LBFGS | 很快 | 简单 | 小型网络、物理优化问题 |

### 学习率推荐

| 优化器 | 典型学习率范围 |
|--------|--------------|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam/AdamW | 1e-4 - 1e-3 |
| RMSprop | 1e-3 - 1e-2 |

---

## 5.2.6 参数组

### 为不同层设置不同学习率

```python
# 分层学习率 - 常用于迁移学习
model = torchvision.models.resnet18(pretrained=True)

optimizer = optim.Adam([
    # 预训练层使用较小学习率
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    # 新添加的分类头使用较大学习率
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=1e-4)  # 默认学习率
```

### 冻结参数

```python
# 冻结特定层
for param in model.layer1.parameters():
    param.requires_grad = False

# 只将需要梯度的参数传给优化器
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

### 动态修改学习率

```python
# 方法1：直接修改
for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr

# 方法2：使用学习率调度器（下一章详述）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

---

## 5.2.7 梯度处理

### 梯度裁剪

防止梯度爆炸，在 RNN 中尤其重要。

```python
# 方法1：按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法2：按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# 完整训练步骤
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 梯度累积

当显存不足以容纳大批量时使用。

```python
accumulation_steps = 4  # 累积4次梯度
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5.2.8 物理优化问题示例

### 使用 LBFGS 求解变分问题

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def variational_optimization():
    """
    使用神经网络求解变分问题
    
    最小化泛函：J[y] = ∫₀¹ (y'² + y²) dx
    边界条件：y(0) = 0, y(1) = 1
    
    解析解：y(x) = sinh(x) / sinh(1)
    """
    
    # 简单的神经网络表示 y(x)
    class VariationalNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # 自动满足边界条件：y(0)=0, y(1)=1
            # y(x) = x + x(1-x) * NN(x)
            nn_out = self.net(x)
            return x + x * (1 - x) * nn_out
    
    model = VariationalNet()
    
    # 积分点
    x = torch.linspace(0, 1, 100, requires_grad=True).reshape(-1, 1)
    
    # 使用 LBFGS 优化
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
    
    def closure():
        optimizer.zero_grad()
        
        # 计算 y 和 y'
        y = model(x)
        y_x = torch.autograd.grad(
            y.sum(), x, create_graph=True
        )[0]
        
        # 泛函：∫(y'² + y²)dx
        integrand = y_x**2 + y**2
        J = integrand.mean()  # 蒙特卡洛积分
        
        J.backward()
        return J
    
    # 优化
    for epoch in range(100):
        loss = optimizer.step(closure)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, J = {loss.item():.6f}")
    
    # 比较结果
    with torch.no_grad():
        x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
        y_pred = model(x_test)
        y_exact = torch.sinh(x_test) / torch.sinh(torch.tensor(1.0))
    
    return x_test.numpy(), y_pred.numpy(), y_exact.numpy()

# 运行
x, y_pred, y_exact = variational_optimization()
```

### 能量最小化

```python
def energy_minimization():
    """
    寻找势能最小的粒子构型
    
    N 个带电粒子在二维平面上，
    相互作用势：V(r) = 1/r（库仑势）
    约束：粒子在单位圆内
    """
    N = 10  # 粒子数
    
    # 粒子位置作为可优化参数
    positions = torch.nn.Parameter(torch.randn(N, 2) * 0.3)
    
    def compute_energy(pos):
        """计算总势能"""
        energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r = torch.norm(pos[i] - pos[j])
                energy += 1.0 / (r + 1e-6)  # 库仑势
        
        # 添加约束势：保持粒子在单位圆内
        r2 = (pos**2).sum(dim=1)
        constraint = torch.relu(r2 - 1.0).sum() * 100
        
        return energy + constraint
    
    optimizer = optim.Adam([positions], lr=0.01)
    
    for step in range(1000):
        optimizer.zero_grad()
        energy = compute_energy(positions)
        energy.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"Step {step}, Energy = {energy.item():.4f}")
    
    return positions.detach()

# 运行
final_positions = energy_minimization()
```

---

## 5.2.9 常见问题与调试

### 梯度消失/爆炸检测

```python
def check_gradients(model):
    """检查梯度健康状况"""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            if param_norm > 100:
                print(f"⚠️ 梯度过大: {name}, norm = {param_norm:.2f}")
            elif param_norm < 1e-7:
                print(f"⚠️ 梯度过小: {name}, norm = {param_norm:.2e}")
    
    total_norm = total_norm ** 0.5
    print(f"总梯度范数: {total_norm:.4f}")
    return total_norm
```

### 学习率查找

```python
def find_learning_rate(model, train_loader, criterion, 
                       init_lr=1e-7, final_lr=10, num_steps=100):
    """
    学习率范围测试
    
    找到合适的初始学习率
    """
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    
    # 学习率乘数
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    
    lrs = []
    losses = []
    
    model.train()
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # 增加学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        # 如果损失爆炸，停止
        if loss.item() > 4 * min(losses):
            break
    
    return lrs, losses
```

---

## 🔬 物理视角总结

### 优化器作为动力学系统

| 优化器特性 | 物理意义 |
|-----------|---------|
| 学习率 | 时间步长 $\Delta t$ |
| 动量 | 惯性质量 |
| 权重衰减 | 谐振子势能 |
| 自适应学习率 | 变步长积分 |
| 梯度噪声 | 热涨落 |

### 收敛的物理图像

训练过程可以理解为在损失函数势能面上寻找全局最小值：

1. **初始阶段**：粒子从随机位置开始，快速下降
2. **中间阶段**：在势能面上探索，可能被困在局部极小
3. **收敛阶段**：在极小值附近振荡，最终稳定

动量和随机噪声帮助粒子跳出局部极小，找到更好的解。

---

## 📝 练习

1. 比较 SGD 和 Adam 在同一问题上的收敛速度
2. 实现学习率查找，为你的模型找到最佳初始学习率
3. 使用 LBFGS 求解一个简单的变分问题

---

## ⏭️ 下一节

下一节我们将学习 [训练循环](./03_training_loop.md)，将数据加载和优化器整合到完整的训练流程中。

