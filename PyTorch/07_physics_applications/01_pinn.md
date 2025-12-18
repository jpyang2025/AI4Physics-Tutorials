# 7.1 物理信息神经网络（PINN）

## 📖 概述

物理信息神经网络（Physics-Informed Neural Networks, PINN）是一种将物理定律嵌入神经网络的方法，用于求解正问题（给定方程求解）和逆问题（从数据推断参数）。

## 🎯 学习目标

- 理解 PINN 的核心思想
- 掌握 PINN 求解 ODE 和 PDE 的方法
- 实现边界条件和初始条件的处理
- 应用 PINN 解决实际物理问题

---

## 7.1.1 PINN 基本原理

### 核心思想

PINN 利用神经网络作为微分方程解的近似，并通过**自动微分**计算导数，将**PDE 残差**作为损失函数的一部分。

```
输入 (x, t) → 神经网络 → 输出 u(x,t)
                ↓
        自动微分计算 ∂u/∂t, ∂u/∂x, ∂²u/∂x², ...
                ↓
        PDE 残差 = PDE(u, ∂u/∂t, ∂²u/∂x², ...) 
                ↓
        最小化残差 → 得到满足 PDE 的解
```

### 损失函数结构

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{data}} \mathcal{L}_{\text{data}}$$

- $\mathcal{L}_{\text{PDE}}$：PDE 残差（在内部配点上）
- $\mathcal{L}_{\text{BC}}$：边界条件残差
- $\mathcal{L}_{\text{IC}}$：初始条件残差
- $\mathcal{L}_{\text{data}}$：观测数据拟合（可选）

---

## 7.1.2 简单 ODE 求解

### 例1：一阶 ODE

$$\frac{dy}{dx} = -y, \quad y(0) = 1$$

解析解：$y(x) = e^{-x}$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN_ODE(nn.Module):
    """求解一阶 ODE 的 PINN"""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def solve_first_order_ode():
    """求解 dy/dx = -y, y(0) = 1"""
    
    model = PINN_ODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 内部配点
    x_interior = torch.linspace(0, 5, 100, requires_grad=True).reshape(-1, 1)
    
    # 边界点
    x_bc = torch.zeros(1, 1)
    y_bc = torch.ones(1, 1)  # y(0) = 1
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        # ODE 残差
        y = model(x_interior)
        dy_dx = torch.autograd.grad(
            y, x_interior, 
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        residual = dy_dx + y  # dy/dx + y = 0
        loss_ode = (residual ** 2).mean()
        
        # 边界条件
        y_pred_bc = model(x_bc)
        loss_bc = (y_pred_bc - y_bc) ** 2
        
        # 总损失
        loss = loss_ode + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # 验证
    x_test = torch.linspace(0, 5, 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test)
    y_exact = torch.exp(-x_test)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='PINN', linewidth=2)
    plt.plot(x_test.numpy(), y_exact.numpy(), 'b--', label='解析解', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('ODE 解')
    
    plt.subplot(1, 2, 2)
    error = (y_pred - y_exact).abs()
    plt.plot(x_test.numpy(), error.numpy())
    plt.xlabel('x')
    plt.ylabel('|误差|')
    plt.title('绝对误差')
    
    plt.tight_layout()
    plt.show()

solve_first_order_ode()
```

### 例2：二阶 ODE（简谐振子）

$$\frac{d^2 y}{dx^2} + \omega^2 y = 0, \quad y(0) = 1, \quad y'(0) = 0$$

解析解：$y(x) = \cos(\omega x)$

```python
class PINN_SHO(nn.Module):
    """简谐振子 PINN"""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def solve_harmonic_oscillator(omega=2.0):
    """求解 y'' + ω²y = 0"""
    
    model = PINN_SHO()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 内部配点
    x_interior = torch.linspace(0.01, 2*torch.pi, 200, requires_grad=True).reshape(-1, 1)
    
    # 初始条件点
    x_ic = torch.zeros(1, 1, requires_grad=True)
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        # ODE 残差
        y = model(x_interior)
        dy_dx = torch.autograd.grad(
            y, x_interior,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        d2y_dx2 = torch.autograd.grad(
            dy_dx, x_interior,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True
        )[0]
        
        residual = d2y_dx2 + omega**2 * y
        loss_ode = (residual ** 2).mean()
        
        # 初始条件：y(0) = 1
        y_ic = model(x_ic)
        loss_ic1 = (y_ic - 1.0) ** 2
        
        # 初始条件：y'(0) = 0
        dy_ic = torch.autograd.grad(
            y_ic, x_ic,
            grad_outputs=torch.ones_like(y_ic),
            create_graph=True
        )[0]
        loss_ic2 = dy_ic ** 2
        
        # 总损失
        loss = loss_ode + 100 * (loss_ic1 + loss_ic2)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 7.1.3 偏微分方程求解

### 热传导方程

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

初始条件：$u(x, 0) = \sin(\pi x)$
边界条件：$u(0, t) = u(1, t) = 0$

```python
class PINN_Heat(nn.Module):
    """热传导方程 PINN"""
    
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def solve_heat_equation(alpha=0.1):
    """
    求解热传导方程
    
    ∂u/∂t = α ∂²u/∂x²
    """
    
    model = PINN_Heat()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 配点
    n_interior = 2000
    n_bc = 100
    n_ic = 100
    
    for epoch in range(10000):
        optimizer.zero_grad()
        
        # 内部配点（随机采样）
        x_int = torch.rand(n_interior, 1, requires_grad=True)
        t_int = torch.rand(n_interior, 1, requires_grad=True) * 0.5
        
        u = model(x_int, t_int)
        
        # 计算偏导数
        u_t = torch.autograd.grad(
            u, t_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x_int, grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        # PDE 残差
        residual = u_t - alpha * u_xx
        loss_pde = (residual ** 2).mean()
        
        # 初始条件：u(x, 0) = sin(πx)
        x_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)
        u_ic_pred = model(x_ic, t_ic)
        u_ic_true = torch.sin(torch.pi * x_ic)
        loss_ic = ((u_ic_pred - u_ic_true) ** 2).mean()
        
        # 边界条件：u(0,t) = u(1,t) = 0
        t_bc = torch.rand(n_bc, 1) * 0.5
        
        x_bc_left = torch.zeros(n_bc, 1)
        u_bc_left = model(x_bc_left, t_bc)
        
        x_bc_right = torch.ones(n_bc, 1)
        u_bc_right = model(x_bc_right, t_bc)
        
        loss_bc = (u_bc_left ** 2).mean() + (u_bc_right ** 2).mean()
        
        # 总损失
        loss = loss_pde + 10 * loss_ic + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1}: PDE={loss_pde.item():.6f}, "
                  f"IC={loss_ic.item():.6f}, BC={loss_bc.item():.6f}")
    
    return model
```

### 波动方程

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

```python
class PINN_Wave(nn.Module):
    """波动方程 PINN"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def solve_wave_equation(c=1.0):
    """
    求解波动方程
    
    ∂²u/∂t² = c² ∂²u/∂x²
    初始条件：u(x,0) = sin(πx), ∂u/∂t(x,0) = 0
    边界条件：u(0,t) = u(1,t) = 0
    """
    
    model = PINN_Wave()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(15000):
        optimizer.zero_grad()
        
        # 内部配点
        x_int = torch.rand(1000, 1, requires_grad=True)
        t_int = torch.rand(1000, 1, requires_grad=True) * 2.0
        
        u = model(x_int, t_int)
        
        # 计算二阶偏导
        u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_int, torch.ones_like(u_t), create_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        
        # PDE 残差
        residual = u_tt - c**2 * u_xx
        loss_pde = (residual ** 2).mean()
        
        # 初始条件
        x_ic = torch.rand(100, 1, requires_grad=True)
        t_ic = torch.zeros(100, 1, requires_grad=True)
        
        u_ic = model(x_ic, t_ic)
        u_ic_true = torch.sin(torch.pi * x_ic)
        loss_ic1 = ((u_ic - u_ic_true) ** 2).mean()
        
        # ∂u/∂t(x,0) = 0
        u_t_ic = torch.autograd.grad(u_ic, t_ic, torch.ones_like(u_ic), create_graph=True)[0]
        loss_ic2 = (u_t_ic ** 2).mean()
        
        # 边界条件
        t_bc = torch.rand(100, 1) * 2.0
        u_left = model(torch.zeros(100, 1), t_bc)
        u_right = model(torch.ones(100, 1), t_bc)
        loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()
        
        loss = loss_pde + 10 * (loss_ic1 + loss_ic2) + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 3000 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 7.1.4 硬约束边界条件

### 构造自动满足边界条件的解

通过巧妙的网络输出变换，可以让解**自动满足边界条件**。

```python
class PINN_HardBC(nn.Module):
    """
    硬约束边界条件的 PINN
    
    对于 u(0) = a, u(1) = b 的 Dirichlet 边界条件，
    构造：u(x) = a(1-x) + bx + x(1-x)·NN(x)
    """
    
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = a
        self.b = b
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # 基础函数满足边界条件
        base = self.a * (1 - x) + self.b * x
        # 修正项在边界为零
        correction = x * (1 - x) * self.net(x)
        return base + correction


class PINN_TimeDependent_HardBC(nn.Module):
    """
    时间相关问题的硬约束
    
    满足：
    - u(x, 0) = f(x) (初始条件)
    - u(0, t) = g0(t), u(1, t) = g1(t) (边界条件)
    """
    
    def __init__(self, f_init, g0_bc, g1_bc):
        super().__init__()
        self.f_init = f_init    # 初始条件函数
        self.g0_bc = g0_bc      # 左边界函数
        self.g1_bc = g1_bc      # 右边界函数
        
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        nn_out = self.net(inputs)
        
        # 满足初始条件的项
        u_init = self.f_init(x)
        
        # 满足边界条件的插值
        u_bc = (1 - x) * self.g0_bc(t) + x * self.g1_bc(t)
        
        # 组合（t=0 时等于初始条件，x=0,1 时等于边界条件）
        # 修正项在 t=0 和 x=0,1 处为零
        correction = t * x * (1 - x) * nn_out
        
        # 需要更复杂的构造来同时满足 IC 和 BC
        # 这里简化处理
        return u_init * torch.exp(-t) + (1 - torch.exp(-t)) * u_bc + correction
```

---

## 7.1.5 逆问题：参数估计

### 从数据推断未知参数

```python
class PINN_Inverse(nn.Module):
    """
    逆问题 PINN：从数据推断未知参数
    
    例如：从热传导数据推断热扩散系数 α
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # 未知参数（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def solve_inverse_problem():
    """
    从观测数据推断热扩散系数
    
    真实 α = 0.1
    """
    alpha_true = 0.1
    
    # 生成"观测"数据（解析解）
    def analytical_solution(x, t, alpha):
        return torch.sin(torch.pi * x) * torch.exp(-alpha * torch.pi**2 * t)
    
    # 观测点
    n_obs = 50
    x_obs = torch.rand(n_obs, 1)
    t_obs = torch.rand(n_obs, 1) * 0.5
    u_obs = analytical_solution(x_obs, t_obs, alpha_true)
    u_obs = u_obs + 0.01 * torch.randn_like(u_obs)  # 添加噪声
    
    model = PINN_Inverse()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    alpha_history = []
    
    for epoch in range(10000):
        optimizer.zero_grad()
        
        # 数据拟合损失
        u_pred_obs = model(x_obs, t_obs)
        loss_data = ((u_pred_obs - u_obs) ** 2).mean()
        
        # PDE 残差
        x_int = torch.rand(500, 1, requires_grad=True)
        t_int = torch.rand(500, 1, requires_grad=True) * 0.5
        
        u = model(x_int, t_int)
        u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_t - model.alpha * u_xx
        loss_pde = (residual ** 2).mean()
        
        loss = loss_data + 0.1 * loss_pde
        
        loss.backward()
        optimizer.step()
        
        # 确保 α > 0
        with torch.no_grad():
            model.alpha.clamp_(min=0.001)
        
        alpha_history.append(model.alpha.item())
        
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1}: α = {model.alpha.item():.4f} "
                  f"(真实值: {alpha_true})")
    
    print(f"\n最终估计: α = {model.alpha.item():.4f}")
    print(f"相对误差: {abs(model.alpha.item() - alpha_true) / alpha_true * 100:.2f}%")
    
    return model, alpha_history
```

---

## 7.1.6 Burgers 方程

非线性 PDE 的经典例子：

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

```python
class PINN_Burgers(nn.Module):
    """Burgers 方程 PINN"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def solve_burgers_equation(nu=0.01):
    """
    求解 Burgers 方程
    
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    初始条件：u(x,0) = -sin(πx)
    边界条件：u(-1,t) = u(1,t) = 0
    """
    
    model = PINN_Burgers()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20000):
        optimizer.zero_grad()
        
        # 内部配点
        x_int = torch.rand(2000, 1, requires_grad=True) * 2 - 1  # [-1, 1]
        t_int = torch.rand(2000, 1, requires_grad=True)
        
        u = model(x_int, t_int)
        
        u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        
        # Burgers 方程残差
        residual = u_t + u * u_x - nu * u_xx
        loss_pde = (residual ** 2).mean()
        
        # 初始条件
        x_ic = torch.rand(200, 1) * 2 - 1
        t_ic = torch.zeros(200, 1)
        u_ic = model(x_ic, t_ic)
        u_ic_true = -torch.sin(torch.pi * x_ic)
        loss_ic = ((u_ic - u_ic_true) ** 2).mean()
        
        # 边界条件
        t_bc = torch.rand(100, 1)
        u_left = model(-torch.ones(100, 1), t_bc)
        u_right = model(torch.ones(100, 1), t_bc)
        loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()
        
        loss = loss_pde + 10 * loss_ic + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 4000 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 7.1.7 Navier-Stokes 方程

流体力学的基本方程：

```python
class PINN_NavierStokes(nn.Module):
    """
    2D 不可压缩 Navier-Stokes 方程
    
    ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u
    ∇·u = 0
    """
    
    def __init__(self):
        super().__init__()
        # 共享的特征提取器
        self.shared = nn.Sequential(
            nn.Linear(3, 64),  # (x, y, t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # 分别输出 u, v, p
        self.u_head = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.v_head = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.p_head = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
    
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        features = self.shared(inputs)
        
        u = self.u_head(features)
        v = self.v_head(features)
        p = self.p_head(features)
        
        return u, v, p


def navier_stokes_residual(model, x, y, t, nu=0.01, rho=1.0):
    """计算 Navier-Stokes 残差"""
    
    u, v, p = model(x, y, t)
    
    # 一阶导数
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    # 二阶导数
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    
    # 动量方程残差
    res_u = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    
    # 连续性方程残差
    res_cont = u_x + v_y
    
    return res_u, res_v, res_cont
```

---

## 7.1.8 训练技巧

### 自适应权重

```python
class AdaptiveLossWeights:
    """自适应损失权重"""
    
    def __init__(self, n_losses, learning_rate=0.01):
        self.log_weights = torch.zeros(n_losses, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_weights], lr=learning_rate)
    
    def get_weights(self):
        return torch.exp(-self.log_weights)
    
    def update(self, losses):
        """根据损失更新权重"""
        self.optimizer.zero_grad()
        
        # 损失：-log(w) + w * L
        total = 0
        for log_w, loss in zip(self.log_weights, losses):
            total += -log_w + torch.exp(-log_w) * loss.detach()
        
        total.backward()
        self.optimizer.step()
```

### 采样策略

```python
def residual_based_sampling(model, x_range, t_range, n_samples, 
                             pde_residual_fn):
    """
    基于残差的自适应采样
    
    在残差大的区域采样更多点
    """
    # 粗采样评估残差
    n_coarse = 1000
    x_coarse = torch.rand(n_coarse, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t_coarse = torch.rand(n_coarse, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x_coarse.requires_grad = True
    t_coarse.requires_grad = True
    
    with torch.no_grad():
        residual = pde_residual_fn(model, x_coarse, t_coarse)
        weights = residual.abs().squeeze()
        weights = weights / weights.sum()
    
    # 根据权重采样
    indices = torch.multinomial(weights, n_samples, replacement=True)
    
    x_refined = x_coarse[indices].clone().detach().requires_grad_(True)
    t_refined = t_coarse[indices].clone().detach().requires_grad_(True)
    
    return x_refined, t_refined
```

---

## 🔬 物理视角总结

### PINN 的优势

| 方面 | 传统数值方法 | PINN |
|------|-------------|------|
| 网格 | 需要离散化 | 无网格 |
| 高维问题 | 维度灾难 | 相对容易 |
| 逆问题 | 需要特殊处理 | 自然融合 |
| 噪声数据 | 敏感 | 有正则化效果 |

### 局限性

- 训练可能困难，容易陷入局部最优
- 对于强非线性问题精度可能不足
- 超参数（权重）调节需要经验

---

## 📝 练习

1. 用 PINN 求解泊松方程 $\nabla^2 u = f(x,y)$
2. 实现一个 PINN 求解薛定谔方程的时间演化
3. 尝试从含噪声数据中恢复 PDE 的未知参数

---

## ⏭️ 下一节

下一节我们将学习 [分子动力学模拟](./02_molecular_dynamics.md)，了解如何用神经网络构建势函数。

