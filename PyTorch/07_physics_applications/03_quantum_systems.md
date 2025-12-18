# 7.3 量子系统模拟

## 📖 概述

神经网络在量子力学问题中有着广泛应用，包括求解薛定谔方程、表示波函数、进行变分优化等。本节介绍如何用神经网络处理量子系统。

## 🎯 学习目标

- 理解变分原理与神经网络的结合
- 使用神经网络表示波函数
- 求解定态薛定谔方程
- 了解量子多体问题的神经网络方法

---

## 7.3.1 变分原理

### 基本概念

变分原理是量子力学中求解基态的核心方法：

$$E[\psi] = \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} \geq E_0$$

对于任意试探波函数 $\psi$，能量期望值总是大于等于基态能量 $E_0$。

### 神经网络波函数

用神经网络参数化波函数：

$$\psi_\theta(\mathbf{r}) = \text{NN}_\theta(\mathbf{r})$$

然后最小化能量期望值找到最优参数 $\theta^*$。

---

## 7.3.2 一维无限深势阱

### 问题描述

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi, \quad 0 < x < L$$

边界条件：$\psi(0) = \psi(L) = 0$

解析解：$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)$，$E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class WavefunctionNN(nn.Module):
    """
    神经网络波函数
    
    自动满足边界条件 ψ(0) = ψ(L) = 0
    """
    
    def __init__(self, L=1.0, hidden_dim=32):
        super().__init__()
        self.L = L
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # 满足边界条件：ψ = x(L-x) * NN(x)
        return x * (self.L - x) * self.net(x)


def solve_infinite_well():
    """求解一维无限深势阱"""
    
    L = 1.0  # 势阱宽度
    hbar = 1.0
    m = 1.0
    
    model = WavefunctionNN(L)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 积分点（用于计算期望值）
    x = torch.linspace(0.001, L - 0.001, 100, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        # 波函数
        psi = model(x)
        
        # 一阶导数
        dpsi_dx = torch.autograd.grad(
            psi, x, grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]
        
        # 二阶导数
        d2psi_dx2 = torch.autograd.grad(
            dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0]
        
        # 归一化
        norm = (psi ** 2).sum() * dx
        psi_normalized = psi / torch.sqrt(norm)
        d2psi_normalized = d2psi_dx2 / torch.sqrt(norm)
        
        # 动能期望值 T = -ℏ²/(2m) ∫ψ* d²ψ/dx² dx
        # 对于实波函数 T = -ℏ²/(2m) ∫ψ d²ψ/dx² dx
        kinetic_energy = -hbar**2 / (2 * m) * (psi_normalized * d2psi_normalized).sum() * dx
        
        # 势能为零（势阱内）
        energy = kinetic_energy
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
        
        if (epoch + 1) % 500 == 0:
            E_exact = np.pi**2 * hbar**2 / (2 * m * L**2)
            print(f"Epoch {epoch+1}: E = {energy.item():.6f}, "
                  f"精确值 = {E_exact:.6f}")
    
    # 可视化
    model.eval()
    x_plot = torch.linspace(0, L, 100).reshape(-1, 1)
    
    with torch.no_grad():
        psi_pred = model(x_plot)
        # 归一化
        norm = (psi_pred ** 2).sum() * (L / 100)
        psi_pred = psi_pred / torch.sqrt(norm)
    
    # 解析解（基态 n=1）
    psi_exact = np.sqrt(2 / L) * np.sin(np.pi * x_plot.numpy() / L)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(x_plot.numpy(), psi_pred.numpy(), 'r-', label='NN', linewidth=2)
    axes[0].plot(x_plot.numpy(), psi_exact, 'b--', label='解析解', linewidth=2)
    axes[0].plot(x_plot.numpy(), -psi_exact, 'b--', alpha=0.3)  # 相位不定
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ψ(x)')
    axes[0].set_title('波函数')
    axes[0].legend()
    
    axes[1].plot(energy_history)
    E_exact = np.pi**2 / 2
    axes[1].axhline(y=E_exact, color='r', linestyle='--', label=f'精确值 = {E_exact:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('能量收敛')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('infinite_well.png', dpi=150)
    plt.show()
    
    return model

solve_infinite_well()
```

---

## 7.3.3 谐振子

### 量子谐振子

$$\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2$$

基态解析解：$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-\frac{m\omega x^2}{2\hbar}}$，$E_0 = \frac{1}{2}\hbar\omega$

```python
class HarmonicOscillatorNN(nn.Module):
    """量子谐振子神经网络波函数"""
    
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
        # 使用高斯包络确保波函数在无穷远衰减
        envelope = torch.exp(-0.5 * x ** 2)
        return envelope * self.net(x)


def solve_harmonic_oscillator():
    """求解量子谐振子"""
    
    # 单位制：ℏ = m = ω = 1
    hbar = 1.0
    m = 1.0
    omega = 1.0
    
    model = HarmonicOscillatorNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 积分范围（波函数在远处衰减）
    x = torch.linspace(-5, 5, 200, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        psi = model(x)
        
        # 计算导数
        dpsi = torch.autograd.grad(
            psi, x, torch.ones_like(psi), create_graph=True
        )[0]
        d2psi = torch.autograd.grad(
            dpsi, x, torch.ones_like(dpsi), create_graph=True
        )[0]
        
        # 归一化
        norm = (psi ** 2).sum() * dx
        psi_n = psi / torch.sqrt(norm)
        d2psi_n = d2psi / torch.sqrt(norm)
        
        # 动能
        T = -hbar**2 / (2 * m) * (psi_n * d2psi_n).sum() * dx
        
        # 势能
        V = 0.5 * m * omega**2 * (x**2 * psi_n**2).sum() * dx
        
        energy = T + V
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: E = {energy.item():.6f}, 精确值 = 0.5")
    
    # 可视化
    model.eval()
    x_plot = torch.linspace(-5, 5, 200).reshape(-1, 1)
    
    with torch.no_grad():
        psi_pred = model(x_plot)
        norm = (psi_pred ** 2).sum() * 0.05
        psi_pred = psi_pred / torch.sqrt(norm)
    
    # 解析解
    psi_exact = (1 / np.pi) ** 0.25 * np.exp(-0.5 * x_plot.numpy() ** 2)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_plot.numpy(), psi_pred.numpy(), 'r-', label='NN', linewidth=2)
    plt.plot(x_plot.numpy(), psi_exact, 'b--', label='解析解', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('ψ(x)')
    plt.title('谐振子基态波函数')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(energy_history)
    plt.axhline(y=0.5, color='r', linestyle='--', label='E₀ = 0.5')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.title('能量收敛')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator.png', dpi=150)
    plt.show()
    
    return model

solve_harmonic_oscillator()
```

---

## 7.3.4 氢原子（径向波函数）

### 径向薛定谔方程

$$-\frac{\hbar^2}{2m}\left[\frac{d^2}{dr^2} + \frac{2}{r}\frac{d}{dr} - \frac{l(l+1)}{r^2}\right]R(r) - \frac{e^2}{4\pi\epsilon_0 r}R(r) = ER(r)$$

对于 $l=0$（s 轨道），基态解析解：$R_{1s}(r) = 2\left(\frac{1}{a_0}\right)^{3/2}e^{-r/a_0}$

```python
class HydrogenNN(nn.Module):
    """氢原子波函数"""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, r):
        # 确保正确的渐近行为：
        # r→0: 有限
        # r→∞: 指数衰减
        envelope = torch.exp(-r)
        return envelope * self.net(r)


def solve_hydrogen_atom():
    """求解氢原子基态（原子单位）"""
    
    # 原子单位：ℏ = m_e = e² = 4πε₀ = 1
    
    model = HydrogenNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 径向坐标
    r = torch.linspace(0.01, 15, 200, requires_grad=True).reshape(-1, 1)
    dr = r[1, 0] - r[0, 0]
    
    energy_history = []
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        R = model(r)
        
        # 计算导数
        dR = torch.autograd.grad(
            R, r, torch.ones_like(R), create_graph=True
        )[0]
        d2R = torch.autograd.grad(
            dR, r, torch.ones_like(dR), create_graph=True
        )[0]
        
        # 归一化（径向波函数：∫|R|² r² dr = 1）
        norm = (R**2 * r**2).sum() * dr
        R_n = R / torch.sqrt(norm)
        dR_n = dR / torch.sqrt(norm)
        d2R_n = d2R / torch.sqrt(norm)
        
        # 动能（l=0）
        # T = -1/2 ∫R* (d²R/dr² + 2/r dR/dr) r² dr
        T = -0.5 * ((d2R_n + 2/r * dR_n) * R_n * r**2).sum() * dr
        
        # 势能
        # V = -∫|R|² (1/r) r² dr = -∫|R|² r dr
        V = -(R_n**2 * r).sum() * dr
        
        energy = T + V
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: E = {energy.item():.6f}, 精确值 = -0.5")
    
    # 可视化
    model.eval()
    r_plot = torch.linspace(0.01, 10, 200).reshape(-1, 1)
    
    with torch.no_grad():
        R_pred = model(r_plot)
        norm = (R_pred**2 * r_plot**2).sum() * 0.05
        R_pred = R_pred / torch.sqrt(norm)
    
    # 解析解（原子单位）
    R_exact = 2 * np.exp(-r_plot.numpy())
    R_exact = R_exact / np.sqrt((R_exact**2 * r_plot.numpy()**2).sum() * 0.05)
    
    # 径向概率密度
    P_pred = (R_pred.numpy()**2 * r_plot.numpy()**2).flatten()
    P_exact = (R_exact**2 * r_plot.numpy()**2).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(r_plot.numpy(), R_pred.numpy(), 'r-', label='NN', linewidth=2)
    axes[0].plot(r_plot.numpy(), R_exact, 'b--', label='解析解', linewidth=2)
    axes[0].set_xlabel('r (a₀)')
    axes[0].set_ylabel('R(r)')
    axes[0].set_title('径向波函数')
    axes[0].legend()
    
    axes[1].plot(r_plot.numpy(), P_pred, 'r-', label='NN', linewidth=2)
    axes[1].plot(r_plot.numpy(), P_exact, 'b--', label='解析解', linewidth=2)
    axes[1].set_xlabel('r (a₀)')
    axes[1].set_ylabel('r²|R(r)|²')
    axes[1].set_title('径向概率密度')
    axes[1].legend()
    
    axes[2].plot(energy_history)
    axes[2].axhline(y=-0.5, color='r', linestyle='--', label='E = -0.5 Hartree')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Energy (Hartree)')
    axes[2].set_title('能量收敛')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('hydrogen_atom.png', dpi=150)
    plt.show()
    
    return model

solve_hydrogen_atom()
```

---

## 7.3.5 变分蒙特卡洛（VMC）

### 蒙特卡洛积分

对于高维系统，使用蒙特卡洛方法计算期望值：

$$E[\psi] = \frac{\int \psi^*(\mathbf{r}) \hat{H} \psi(\mathbf{r}) d\mathbf{r}}{\int |\psi(\mathbf{r})|^2 d\mathbf{r}} = \int |\psi(\mathbf{r})|^2 E_L(\mathbf{r}) d\mathbf{r}$$

其中局域能量 $E_L(\mathbf{r}) = \frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}$

```python
class VMCWavefunction(nn.Module):
    """变分蒙特卡洛波函数"""
    
    def __init__(self, n_particles, dim=3, hidden_dim=64):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        
        input_dim = n_particles * dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, r):
        """
        Args:
            r: [batch, n_particles * dim]
        
        Returns:
            log|ψ|: [batch, 1]
        """
        return self.net(r)
    
    def log_prob(self, r):
        """log|ψ|²"""
        return 2 * self.forward(r)


def local_energy(wavefunction, r, potential_fn):
    """
    计算局域能量
    
    E_L = -0.5 ∇²ψ/ψ + V
    
    使用自动微分计算拉普拉斯算子
    """
    r = r.requires_grad_(True)
    
    log_psi = wavefunction(r)
    
    # ∇ log ψ
    grad_log_psi = torch.autograd.grad(
        log_psi.sum(), r, create_graph=True
    )[0]
    
    # ∇² log ψ = Tr(∇∇ log ψ)
    # 使用恒等式：∇²ψ/ψ = (∇ log ψ)² + ∇² log ψ
    laplacian_log_psi = 0
    for i in range(r.shape[1]):
        grad_i = torch.autograd.grad(
            grad_log_psi[:, i].sum(), r, create_graph=True
        )[0][:, i]
        laplacian_log_psi += grad_i
    
    # 动能
    kinetic = -0.5 * ((grad_log_psi ** 2).sum(dim=1) + laplacian_log_psi)
    
    # 势能
    potential = potential_fn(r)
    
    return kinetic + potential


def metropolis_sampling(wavefunction, n_samples, n_particles, dim=3,
                         step_size=0.5, n_warmup=1000):
    """
    Metropolis-Hastings 采样
    """
    r = torch.randn(1, n_particles * dim)
    samples = []
    
    n_accepted = 0
    n_total = n_warmup + n_samples
    
    for i in range(n_total):
        # 提议新位置
        r_new = r + step_size * torch.randn_like(r)
        
        # 接受概率
        log_prob_old = wavefunction.log_prob(r)
        log_prob_new = wavefunction.log_prob(r_new)
        
        accept_prob = torch.exp(log_prob_new - log_prob_old).item()
        
        if np.random.random() < min(1, accept_prob):
            r = r_new
            n_accepted += 1
        
        if i >= n_warmup:
            samples.append(r.clone())
    
    acceptance_rate = n_accepted / n_total
    return torch.cat(samples, dim=0), acceptance_rate


def vmc_training(wavefunction, potential_fn, n_particles, dim=3,
                  n_epochs=1000, n_samples=500):
    """
    VMC 训练
    """
    optimizer = torch.optim.Adam(wavefunction.parameters(), lr=0.01)
    
    energy_history = []
    
    for epoch in range(n_epochs):
        # 采样
        samples, acc_rate = metropolis_sampling(
            wavefunction, n_samples, n_particles, dim
        )
        
        optimizer.zero_grad()
        
        # 计算局域能量
        E_L = local_energy(wavefunction, samples, potential_fn)
        
        # 能量期望值
        E_mean = E_L.mean()
        
        # 梯度估计（REINFORCE 风格）
        log_psi = wavefunction(samples)
        loss = (2 * log_psi.squeeze() * (E_L - E_mean).detach()).mean()
        
        loss.backward()
        optimizer.step()
        
        energy_history.append(E_mean.item())
        
        if (epoch + 1) % 100 == 0:
            E_std = E_L.std().item()
            print(f"Epoch {epoch+1}: E = {E_mean.item():.4f} ± {E_std:.4f}, "
                  f"Acc = {acc_rate:.2%}")
    
    return energy_history
```

---

## 7.3.6 激发态求解

### 正交化方法

为了求解激发态，需要让新的波函数与已知低能态正交。

```python
class ExcitedStateNN(nn.Module):
    """激发态波函数"""
    
    def __init__(self, ground_state_model, hidden_dim=32):
        super().__init__()
        self.ground_state = ground_state_model
        self.ground_state.eval()
        
        # 冻结基态模型
        for param in self.ground_state.parameters():
            param.requires_grad = False
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # 某种包络函数
        envelope = torch.exp(-0.5 * x ** 2)
        return envelope * self.net(x)
    
    def orthogonalized(self, x, dx):
        """Gram-Schmidt 正交化"""
        psi_excited = self.forward(x)
        
        with torch.no_grad():
            psi_ground = self.ground_state(x)
            # 归一化基态
            norm_ground = (psi_ground ** 2).sum() * dx
            psi_ground = psi_ground / torch.sqrt(norm_ground)
        
        # 计算重叠
        overlap = (psi_excited * psi_ground).sum() * dx
        
        # 正交化
        psi_orth = psi_excited - overlap * psi_ground
        
        return psi_orth


def solve_first_excited_state(ground_state_model):
    """求解第一激发态"""
    
    model = ExcitedStateNN(ground_state_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.linspace(-6, 6, 200, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        # 正交化的波函数
        psi = model.orthogonalized(x, dx)
        
        # 归一化
        norm = (psi ** 2).sum() * dx
        psi_n = psi / torch.sqrt(norm)
        
        # 计算导数
        dpsi = torch.autograd.grad(
            psi_n.sum(), x, create_graph=True
        )[0]
        d2psi = torch.autograd.grad(
            dpsi.sum(), x, create_graph=True
        )[0]
        
        # 能量（谐振子）
        T = -0.5 * (psi_n * d2psi).sum() * dx
        V = 0.5 * (x**2 * psi_n**2).sum() * dx
        
        energy = T + V
        
        # 添加正交性惩罚
        with torch.no_grad():
            psi_ground = ground_state_model(x)
            norm_ground = (psi_ground ** 2).sum() * dx
            psi_ground = psi_ground / torch.sqrt(norm_ground)
        
        overlap = (psi_n * psi_ground).sum() * dx
        penalty = 100 * overlap ** 2
        
        loss = energy + penalty
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: E = {energy.item():.4f}, "
                  f"Overlap = {overlap.item():.6f}")
    
    return model
```

---

## 7.3.7 二粒子系统

### 两个相互作用的粒子

```python
class TwoParticleWavefunction(nn.Module):
    """两粒子波函数"""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 输入：两个粒子的坐标 (x1, x2)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: [batch, 1]
        
        Returns:
            ψ(x1, x2)
        """
        inputs = torch.cat([x1, x2], dim=1)
        
        # 对称化（玻色子）或反对称化（费米子）
        # 这里实现玻色子对称波函数
        psi_12 = self.net(inputs)
        psi_21 = self.net(torch.cat([x2, x1], dim=1))
        
        return (psi_12 + psi_21) / np.sqrt(2)


def solve_two_particle_harmonic():
    """
    求解两个相互作用粒子在谐振子势中的基态
    
    H = -0.5(d²/dx₁² + d²/dx₂²) + 0.5(x₁² + x₂²) + g·δ(x₁-x₂)
    """
    
    model = TwoParticleWavefunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 二维网格
    n_points = 30
    x1_grid = torch.linspace(-4, 4, n_points)
    x2_grid = torch.linspace(-4, 4, n_points)
    X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing='ij')
    
    x1 = X1.reshape(-1, 1).requires_grad_(True)
    x2 = X2.reshape(-1, 1).requires_grad_(True)
    
    dx = x1_grid[1] - x1_grid[0]
    
    # 相互作用强度
    g = 1.0
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        psi = model(x1, x2)
        
        # 归一化
        norm = (psi ** 2).sum() * dx ** 2
        psi_n = psi / torch.sqrt(norm)
        
        # 动能
        dpsi_dx1 = torch.autograd.grad(psi_n.sum(), x1, create_graph=True)[0]
        d2psi_dx1 = torch.autograd.grad(dpsi_dx1.sum(), x1, create_graph=True)[0]
        
        dpsi_dx2 = torch.autograd.grad(psi_n.sum(), x2, create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx2.sum(), x2, create_graph=True)[0]
        
        T = -0.5 * (psi_n * (d2psi_dx1 + d2psi_dx2)).sum() * dx ** 2
        
        # 谐振子势能
        V_ho = 0.5 * ((x1**2 + x2**2) * psi_n**2).sum() * dx ** 2
        
        # 相互作用（用高斯近似 δ 函数）
        sigma = 0.3
        interaction = g * torch.exp(-(x1 - x2)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        V_int = (interaction * psi_n**2).sum() * dx ** 2
        
        energy = T + V_ho + V_int
        
        energy.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: E = {energy.item():.4f}")
    
    # 可视化
    model.eval()
    with torch.no_grad():
        psi_plot = model(x1, x2).reshape(n_points, n_points)
        psi_plot = psi_plot / torch.sqrt((psi_plot ** 2).sum() * dx ** 2)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(X1.numpy(), X2.numpy(), psi_plot.numpy() ** 2, levels=20, cmap='hot')
    plt.colorbar(label='|ψ(x₁,x₂)|²')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('两粒子基态概率密度')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('two_particle.png', dpi=150)
    plt.show()
    
    return model
```

---

## 7.3.8 神经网络量子态层析

### 从测量数据重构量子态

```python
class QuantumStateTomography(nn.Module):
    """
    量子态层析：从测量数据重构波函数
    """
    
    def __init__(self, n_qubits, hidden_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 实部和虚部分别用神经网络表示
        self.real_net = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.imag_net = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, basis_states):
        """
        Args:
            basis_states: [batch, n_qubits] 基态配置 (0 或 1)
        
        Returns:
            复数波函数系数
        """
        # 转换为 ±1 表示
        spins = 2 * basis_states.float() - 1
        
        real = self.real_net(spins)
        imag = self.imag_net(spins)
        
        return real, imag
    
    def probability(self, basis_states):
        """计算测量概率"""
        real, imag = self.forward(basis_states)
        return real ** 2 + imag ** 2


def quantum_tomography_example():
    """量子态层析示例"""
    
    n_qubits = 2
    
    # 生成"测量数据"（假设真实态是 Bell 态）
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    true_probs = torch.tensor([0.5, 0, 0, 0.5])  # |00⟩, |01⟩, |10⟩, |11⟩
    
    # 采样模拟测量
    n_measurements = 1000
    measurements = torch.multinomial(true_probs, n_measurements, replacement=True)
    
    # 转换为基态配置
    basis_configs = torch.tensor([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ], dtype=torch.float32)
    
    # 统计频率
    counts = torch.zeros(4)
    for m in measurements:
        counts[m] += 1
    measured_probs = counts / n_measurements
    
    print(f"测量频率: {measured_probs}")
    print(f"真实概率: {true_probs}")
    
    # 训练模型
    model = QuantumStateTomography(n_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    all_states = basis_configs
    
    for epoch in range(2000):
        optimizer.zero_grad()
        
        probs = model.probability(all_states).squeeze()
        
        # 归一化
        probs_normalized = probs / probs.sum()
        
        # 负对数似然损失
        loss = -torch.sum(measured_probs * torch.log(probs_normalized + 1e-10))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
            print(f"  重构概率: {probs_normalized.detach()}")
    
    return model
```

---

## 🔬 物理视角总结

### 神经网络在量子力学中的优势

| 方法 | 传统方法 | 神经网络方法 |
|------|---------|-------------|
| 基组展开 | 需要选择基组 | 自适应表示 |
| 多体问题 | 指数复杂度 | 多项式参数 |
| 对称性 | 需要显式处理 | 可以学习 |
| 激发态 | 需要正交化 | 可以惩罚重叠 |

### 挑战

1. **符号问题**：费米子系统的反对称性
2. **归一化**：保持波函数归一化
3. **采样**：高效地从 $|\psi|^2$ 采样
4. **优化**：能量景观可能复杂

---

## 📝 练习

1. 用神经网络求解双势阱问题
2. 实现一维氢分子离子 $H_2^+$ 的基态求解
3. 使用 VMC 方法求解氦原子基态

---

## 🎉 总结

恭喜你完成了第7章的学习！你已经掌握了：

- PINN 求解各类微分方程
- 神经网络势函数进行分子动力学模拟
- 变分方法求解量子力学问题

这些方法代表了物理学与机器学习交叉的前沿，正在改变计算物理的面貌。

