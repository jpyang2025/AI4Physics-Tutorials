# 6.2 学习率调度

## 📖 概述

学习率是深度学习中最重要的超参数之一。合适的学习率调度策略可以显著提升模型性能和训练稳定性。

从物理角度看，学习率调度类似于**模拟退火**——通过逐步降低"温度"来找到更好的解。

## 🎯 学习目标

- 理解学习率调度的原理
- 掌握常用的学习率调度器
- 实现自定义调度策略
- 学会使用学习率预热

---

## 6.2.1 学习率的物理意义

### 梯度下降的动力学

梯度下降可以写成：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

将其视为连续时间动力学：

$$\frac{d\theta}{dt} = -\eta \nabla L(\theta)$$

学习率 $\eta$ 相当于：
- **时间步长**：$\eta = \Delta t$
- **系统温度**：影响在能量景观中的探索能力
- **阻尼系数的倒数**：在过阻尼系统中 $\eta \sim 1/\gamma$

### 学习率调度的物理图像

```
                    高学习率                      低学习率
                    (高温)                        (低温)
                    
能量              ╭─╮                           ╭─╮
景观            ╭─╯ ╰─╮                       ╭─╯ ╰─╮
               ╯     ╰─╮                     ╯     ╰─╮
              ╯       ╰─╮   ───────────>    ╯       ╰─╮
             ╯          ╰                  ╯          ╰
            ●←──→●←─→●    探索            ●            稳定
              大幅振荡                     收敛到极小值
```

---

## 6.2.2 PyTorch 学习率调度器

### 基本用法

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CyclicLR, LambdaLR
)

# 创建优化器
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
for epoch in range(100):
    # 训练代码...
    train_one_epoch()
    
    # 更新学习率（每个 epoch 结束时）
    scheduler.step()
    
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

### 常用调度器一览

```python
def visualize_schedulers():
    """可视化各种学习率调度器"""
    import matplotlib.pyplot as plt
    
    epochs = 100
    initial_lr = 0.1
    
    schedulers = {}
    
    # 1. StepLR - 阶梯下降
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['StepLR'] = (opt, StepLR(opt, step_size=30, gamma=0.1))
    
    # 2. MultiStepLR - 多阶段阶梯下降
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['MultiStepLR'] = (opt, MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.1))
    
    # 3. ExponentialLR - 指数衰减
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['ExponentialLR'] = (opt, ExponentialLR(opt, gamma=0.95))
    
    # 4. CosineAnnealingLR - 余弦退火
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['CosineAnnealingLR'] = (opt, CosineAnnealingLR(opt, T_max=epochs))
    
    # 收集学习率
    lrs = {name: [] for name in schedulers}
    
    for epoch in range(epochs):
        for name, (opt, sched) in schedulers.items():
            lrs[name].append(opt.param_groups[0]['lr'])
            sched.step()
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for ax, (name, lr_values) in zip(axes.flat, lrs.items()):
        ax.plot(lr_values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
```

---

## 6.2.3 StepLR 和 MultiStepLR

### StepLR - 固定步长衰减

```python
# 每 step_size 个 epoch，学习率乘以 gamma
scheduler = StepLR(
    optimizer,
    step_size=30,    # 每30个epoch衰减一次
    gamma=0.1        # 衰减系数
)

# 学习率变化：0.1 → 0.01 → 0.001 → ...
# 在 epoch 30, 60, 90, ... 衰减
```

### MultiStepLR - 指定里程碑衰减

```python
# 在指定的 epoch 衰减
scheduler = MultiStepLR(
    optimizer,
    milestones=[50, 75, 90],  # 在这些epoch衰减
    gamma=0.1
)

# 学习率变化：
# epoch 0-49: 0.1
# epoch 50-74: 0.01
# epoch 75-89: 0.001
# epoch 90+: 0.0001
```

---

## 6.2.4 ExponentialLR - 指数衰减

```python
# 每个 epoch 学习率乘以 gamma
scheduler = ExponentialLR(
    optimizer,
    gamma=0.95  # 每 epoch 衰减 5%
)

# lr(t) = lr(0) * gamma^t
```

### 物理类比：放射性衰变

指数衰减对应物理中的放射性衰变：

$$\text{lr}(t) = \text{lr}_0 \cdot e^{-\lambda t}$$

其中 $\gamma = e^{-\lambda}$。

---

## 6.2.5 CosineAnnealingLR - 余弦退火

### 基本余弦退火

```python
# 学习率按余弦函数从初始值衰减到最小值
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,      # 周期长度
    eta_min=1e-6    # 最小学习率
)

# lr(t) = eta_min + 0.5 * (lr_0 - eta_min) * (1 + cos(π * t / T_max))
```

### 带重启的余弦退火

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 周期性重启
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,         # 初始周期长度
    T_mult=2,       # 每次重启后周期倍增
    eta_min=1e-6
)
```

### 物理类比：振荡降温

余弦退火类似于**周期性振荡**叠加**整体降温**：

```
LR
^
│ ╭╮    ╭─╮      ╭───╮          ╭───────╮
│╭╯ ╰──╮╯  ╰────╮╯    ╰────────╮╯        ╰────────
│                                              
└─────────────────────────────────────────────> Epoch
```

---

## 6.2.6 ReduceLROnPlateau - 自适应衰减

根据验证损失自动调整学习率。

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控最小值
    factor=0.1,        # 衰减因子
    patience=10,       # 等待改善的epoch数
    threshold=0.0001,  # 判断改善的阈值
    min_lr=1e-6        # 最小学习率
)

# 训练循环
for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    # 根据验证损失调整学习率
    scheduler.step(val_loss)  # 注意：传入监控指标
```

### 物理类比：反馈控制

ReduceLROnPlateau 类似于**反馈控制系统**：

- 监测系统状态（验证损失）
- 当系统不再改善时，调整控制参数（学习率）

---

## 6.2.7 OneCycleLR - 单周期策略

现代最有效的学习率策略之一。

```python
# 学习率先升后降，完成一个完整周期
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,              # 最大学习率
    epochs=100,              # 总epoch数
    steps_per_epoch=len(train_loader),  # 每epoch的step数
    pct_start=0.3,           # 上升阶段占比
    anneal_strategy='cos',   # 下降策略
    div_factor=25,           # 初始lr = max_lr / div_factor
    final_div_factor=10000   # 最终lr = max_lr / final_div_factor
)

# 注意：OneCycleLR 需要在每个 batch 后调用
for epoch in range(100):
    for batch in train_loader:
        train_step(batch)
        scheduler.step()  # 每个 batch 后更新
```

### 学习率曲线

```
LR
^
│       ╭─────╮
│     ╭─╯     ╰─╮
│   ╭─╯         ╰─╮
│ ╭─╯             ╰─╮
│─╯                 ╰───────
└───────────────────────────> Steps
 ↑               ↑
 warmup         anneal
```

---

## 6.2.8 学习率预热（Warmup）

### 为什么需要预热？

训练初期，网络参数随机初始化，梯度可能很大或方向混乱。预热阶段使用小学习率，让网络"稳定下来"。

### 线性预热

```python
def get_linear_warmup_scheduler(optimizer, warmup_epochs, total_epochs, 
                                 after_scheduler):
    """
    线性预热 + 后续调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    return warmup_scheduler


class WarmupScheduler:
    """预热调度器包装器"""
    
    def __init__(self, optimizer, warmup_epochs, after_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * warmup_factor
        else:
            # 使用后续调度器
            self.after_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# 使用示例
optimizer = optim.Adam(model.parameters(), lr=0.001)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=95)
scheduler = WarmupScheduler(optimizer, warmup_epochs=5, 
                            after_scheduler=cosine_scheduler)
```

### Transformer 风格预热

```python
class TransformerLRScheduler:
    """
    Transformer 论文中的学习率调度
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
    
    def _compute_lr(self):
        step = self.current_step
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )


# 使用
scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)
```

---

## 6.2.9 Cyclic Learning Rate

### CyclicLR

```python
# 学习率在两个边界之间周期性变化
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-4,         # 下界
    max_lr=1e-2,          # 上界
    step_size_up=2000,    # 上升阶段的step数
    step_size_down=2000,  # 下降阶段的step数
    mode='triangular',    # 模式：triangular, triangular2, exp_range
    cycle_momentum=True   # 同步调整动量
)
```

### 物理类比：周期性驱动

周期性学习率类似于**周期性外力驱动**，可以帮助系统跳出局部极小：

$$\eta(t) = \eta_0 + \Delta\eta \cdot \sin(\omega t)$$

---

## 6.2.10 自定义调度器

### LambdaLR

```python
# 使用自定义函数
def custom_lr_lambda(epoch):
    """自定义学习率函数"""
    if epoch < 10:
        return epoch / 10  # 预热
    elif epoch < 50:
        return 1.0  # 保持
    else:
        return 0.1 ** ((epoch - 50) / 50)  # 衰减

scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)
```

### 完全自定义调度器

```python
class CustomScheduler:
    """完全自定义的学习率调度器"""
    
    def __init__(self, optimizer, schedule_fn):
        """
        Args:
            optimizer: 优化器
            schedule_fn: 函数 f(epoch) -> lr_multiplier
        """
        self.optimizer = optimizer
        self.schedule_fn = schedule_fn
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        multiplier = self.schedule_fn(self.current_epoch)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * multiplier
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# 物理启发的调度：模拟退火
def simulated_annealing_schedule(epoch, T0=1.0, T_min=0.01, alpha=0.99):
    """
    模拟退火调度
    T(n) = max(T_min, T0 * alpha^n)
    """
    return max(T_min, T0 * (alpha ** epoch))

scheduler = CustomScheduler(optimizer, simulated_annealing_schedule)
```

---

## 6.2.11 调度器链

### 顺序调度器

```python
from torch.optim.lr_scheduler import SequentialLR

# 组合多个调度器
scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)  # 预热
scheduler2 = CosineAnnealingLR(optimizer, T_max=95)  # 余弦退火

scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler1, scheduler2],
    milestones=[5]  # 在第5个epoch切换
)
```

### 链式调度器

```python
from torch.optim.lr_scheduler import ChainedScheduler

# 同时应用多个调度器（效果相乘）
scheduler = ChainedScheduler([
    ExponentialLR(optimizer, gamma=0.99),  # 基础衰减
    CyclicLR(optimizer, base_lr=0.001, max_lr=0.01)  # 周期性变化
])
```

---

## 6.2.12 学习率查找

### 自动找到最佳学习率

```python
def find_lr(model, train_loader, criterion, optimizer, 
            init_lr=1e-8, final_lr=10, num_steps=100, device='cpu'):
    """
    学习率范围测试
    
    参考：Leslie Smith 的论文 "Cyclical Learning Rates for Training Neural Networks"
    """
    model.train()
    model = model.to(device)
    
    # 保存初始状态
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 学习率乘数
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    # 设置初始学习率
    for pg in optimizer.param_groups:
        pg['lr'] = init_lr
    
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        
        # 检查是否发散
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss:
            print("损失发散，停止搜索")
            break
        
        loss.backward()
        optimizer.step()
        
        # 增加学习率
        for pg in optimizer.param_groups:
            pg['lr'] *= lr_mult
    
    # 恢复初始状态
    model.load_state_dict(initial_state)
    
    # 找到最佳学习率（损失下降最快的点）
    gradients = np.gradient(losses)
    best_idx = np.argmin(gradients)
    suggested_lr = lrs[best_idx]
    
    print(f"建议学习率: {suggested_lr:.2e}")
    
    # 绘图
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(lrs, losses)
    plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'建议: {suggested_lr:.2e}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogx(lrs[:len(gradients)], gradients)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss Gradient')
    plt.title('Loss Gradient')
    
    plt.tight_layout()
    plt.show()
    
    return suggested_lr, lrs, losses
```

---

## 6.2.13 实践建议

### 调度器选择指南

| 场景 | 推荐调度器 |
|------|-----------|
| 基线实验 | StepLR 或 MultiStepLR |
| 追求最佳性能 | CosineAnnealingLR 或 OneCycleLR |
| 训练不稳定 | Warmup + 任意调度器 |
| 自动调整 | ReduceLROnPlateau |
| Transformer | Warmup + Inverse Square Root 或 Cosine |

### 常见错误

```python
# ❌ 错误：忘记在每个 epoch 调用 scheduler.step()
for epoch in range(100):
    train()
    # scheduler.step()  # 忘记调用！

# ✓ 正确
for epoch in range(100):
    train()
    scheduler.step()

# ❌ 错误：ReduceLROnPlateau 忘记传入监控指标
scheduler.step()  # 缺少参数

# ✓ 正确
scheduler.step(val_loss)

# ❌ 错误：OneCycleLR 在 epoch 级别调用
for epoch in range(100):
    train()
    scheduler.step()  # 应该在 batch 级别

# ✓ 正确
for epoch in range(100):
    for batch in train_loader:
        train_step(batch)
        scheduler.step()  # 每个 batch 后调用
```

---

## 🔬 物理视角总结

### 学习率调度的物理对应

| 调度策略 | 物理过程 |
|---------|---------|
| 恒定学习率 | 恒温分子动力学 |
| 阶梯衰减 | 阶梯式降温 |
| 指数衰减 | 指数降温 |
| 余弦退火 | 周期性振荡降温 |
| 周期性学习率 | 周期性外力驱动 |
| 预热 | 缓慢升温后再降温 |

### 模拟退火的启示

最优的学习率调度遵循模拟退火的原则：

1. **初始高温**：探索能量景观
2. **缓慢降温**：逐渐收敛到局部极小
3. **最终低温**：稳定在极小值

降温速率不能太快（可能困在高能态）也不能太慢（浪费计算资源）。

---

## 📝 练习

1. 可视化不同学习率调度器的学习率曲线
2. 使用学习率查找器为你的模型找到最佳学习率
3. 实现一个自定义的模拟退火调度器

---

## ⏭️ 下一节

下一节我们将学习 [模型保存与加载](./03_model_save_load.md)，了解如何正确保存和恢复训练状态。

