# 7.2 分子动力学模拟

## 📖 概述

分子动力学（Molecular Dynamics, MD）是模拟原子和分子运动的计算方法。传统 MD 依赖预定义的势能函数，而**神经网络势函数**可以达到量子力学精度同时保持经典力场的效率。

## 🎯 学习目标

- 理解分子动力学的基本原理
- 掌握神经网络势函数的构建方法
- 实现简单的 MD 模拟
- 了解对称性约束和不变性

---

## 7.2.1 分子动力学基础

### 基本原理

分子动力学求解牛顿运动方程：

$$m_i \frac{d^2 \mathbf{r}_i}{dt^2} = \mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$$

其中 $E$ 是系统的势能，$\mathbf{F}_i$ 是作用在原子 $i$ 上的力。

### 关键组件

```
分子动力学模拟
│
├── 势能函数 E(r₁, r₂, ..., rN)
│   └── 传统：经验力场（Lennard-Jones, EAM, ...）
│   └── 现代：神经网络势函数
│
├── 积分器
│   └── Velocity Verlet
│   └── Leapfrog
│
└── 热浴（可选）
    └── Nosé-Hoover
    └── Langevin
```

---

## 7.2.2 简单势能函数

### Lennard-Jones 势

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Lennard-Jones 势
    
    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    """
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return 4 * epsilon * (sr12 - sr6)


def lennard_jones_force(r, epsilon=1.0, sigma=1.0):
    """
    Lennard-Jones 力
    
    F(r) = -dV/dr = 24ε/r [2(σ/r)¹² - (σ/r)⁶]
    """
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return 24 * epsilon / r * (2 * sr12 - sr6)


# 可视化
r = torch.linspace(0.9, 3.0, 100)
V = lennard_jones_potential(r)
F = lennard_jones_force(r)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(r.numpy(), V.numpy())
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('r/σ')
plt.ylabel('V/ε')
plt.title('Lennard-Jones 势能')
plt.ylim(-1.5, 2)

plt.subplot(1, 2, 2)
plt.plot(r.numpy(), F.numpy())
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('r/σ')
plt.ylabel('F·σ/ε')
plt.title('Lennard-Jones 力')

plt.tight_layout()
plt.show()
```

### 多体系统的势能

```python
def compute_pairwise_energy(positions, energy_func):
    """
    计算成对相互作用的总势能
    
    Args:
        positions: [N, 3] 原子坐标
        energy_func: 势能函数 V(r)
    
    Returns:
        总势能
    """
    N = positions.shape[0]
    total_energy = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = torch.norm(positions[i] - positions[j])
            total_energy += energy_func(r_ij)
    
    return total_energy


def compute_forces_autograd(positions, energy_func):
    """
    使用自动微分计算力
    
    F = -∇E
    """
    positions = positions.clone().requires_grad_(True)
    energy = compute_pairwise_energy(positions, energy_func)
    
    forces = -torch.autograd.grad(energy, positions)[0]
    return forces
```

---

## 7.2.3 神经网络势函数

### 基本架构

```python
class SimpleNNPotential(nn.Module):
    """
    简单的神经网络势函数
    
    直接将原子坐标映射到能量
    注意：这个简单版本没有满足平移/旋转不变性
    """
    
    def __init__(self, n_atoms, hidden_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        
        self.net = nn.Sequential(
            nn.Linear(n_atoms * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, positions):
        """
        Args:
            positions: [batch, n_atoms, 3] 或 [n_atoms, 3]
        
        Returns:
            energy: [batch, 1] 或 [1]
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        batch_size = positions.shape[0]
        flat = positions.view(batch_size, -1)
        
        return self.net(flat)
    
    def compute_forces(self, positions):
        """计算力 F = -∇E"""
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        
        forces = -torch.autograd.grad(
            energy.sum(), positions,
            create_graph=True
        )[0]
        
        return forces
```

### 距离描述符

为了满足平移和旋转不变性，使用**距离矩阵**作为输入。

```python
class DistanceDescriptor(nn.Module):
    """
    基于距离的描述符
    
    将原子坐标转换为距离矩阵
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, positions):
        """
        Args:
            positions: [batch, n_atoms, 3]
        
        Returns:
            distances: [batch, n_atoms, n_atoms]
        """
        # 计算成对距离
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 3]
        distances = torch.norm(diff, dim=-1)  # [B, N, N]
        
        return distances


class InvariantNNPotential(nn.Module):
    """
    满足平移和旋转不变性的神经网络势函数
    """
    
    def __init__(self, n_atoms, hidden_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        self.descriptor = DistanceDescriptor()
        
        # 距离矩阵是对称的，只取上三角部分
        n_pairs = n_atoms * (n_atoms - 1) // 2
        
        self.net = nn.Sequential(
            nn.Linear(n_pairs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, positions):
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        batch_size = positions.shape[0]
        distances = self.descriptor(positions)
        
        # 提取上三角部分（不含对角线）
        triu_indices = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1)
        pair_distances = distances[:, triu_indices[0], triu_indices[1]]
        
        return self.net(pair_distances)
    
    def compute_forces(self, positions):
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        
        forces = -torch.autograd.grad(
            energy.sum(), positions
        )[0]
        
        return forces
```

---

## 7.2.4 对称函数描述符

### Behler-Parrinello 对称函数

```python
class BehlerParrinelloDescriptor(nn.Module):
    """
    Behler-Parrinello 对称函数
    
    G2: 径向对称函数
    G4: 角度对称函数
    """
    
    def __init__(self, r_cut=6.0, n_radial=8, n_angular=8):
        super().__init__()
        self.r_cut = r_cut
        
        # 径向函数参数
        self.eta_radial = nn.Parameter(
            torch.linspace(0.1, 2.0, n_radial), requires_grad=False
        )
        self.rs = nn.Parameter(
            torch.linspace(0.5, r_cut - 0.5, n_radial), requires_grad=False
        )
        
        # 角度函数参数
        self.eta_angular = nn.Parameter(
            torch.linspace(0.1, 1.0, n_angular), requires_grad=False
        )
        self.zeta = nn.Parameter(
            torch.tensor([1.0, 2.0, 4.0, 8.0]), requires_grad=False
        )
        self.lambda_vals = nn.Parameter(
            torch.tensor([-1.0, 1.0]), requires_grad=False
        )
    
    def cutoff_function(self, r):
        """平滑截断函数"""
        return torch.where(
            r < self.r_cut,
            0.5 * (torch.cos(torch.pi * r / self.r_cut) + 1),
            torch.zeros_like(r)
        )
    
    def radial_symmetry_function(self, distances):
        """
        G2 径向对称函数
        
        G2 = Σ exp(-η(r_ij - Rs)²) · fc(r_ij)
        """
        # distances: [N, N]
        N = distances.shape[0]
        fc = self.cutoff_function(distances)
        
        G2 = []
        for eta, rs in zip(self.eta_radial, self.rs):
            g = torch.exp(-eta * (distances - rs)**2) * fc
            # 对每个原子求和（排除自身）
            mask = ~torch.eye(N, dtype=torch.bool, device=distances.device)
            G2.append(g[mask].reshape(N, N-1).sum(dim=1))
        
        return torch.stack(G2, dim=1)  # [N, n_radial]
    
    def forward(self, positions):
        """计算描述符"""
        # 简化版本：只使用径向函数
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(diff, dim=-1)
        
        return self.radial_symmetry_function(distances)


class BPNeuralNetwork(nn.Module):
    """
    Behler-Parrinello 神经网络势
    
    E = Σ NN(G_i)
    
    每个原子的能量贡献由其局部环境决定
    """
    
    def __init__(self, n_descriptors, hidden_dim=32):
        super().__init__()
        self.descriptor = BehlerParrinelloDescriptor(n_radial=n_descriptors)
        
        # 原子能量网络（对每个原子独立应用）
        self.atomic_net = nn.Sequential(
            nn.Linear(n_descriptors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, positions):
        """
        Args:
            positions: [N, 3] 原子坐标
        
        Returns:
            energy: 总能量
        """
        # 计算描述符
        G = self.descriptor(positions)  # [N, n_descriptors]
        
        # 计算原子能量
        atomic_energies = self.atomic_net(G)  # [N, 1]
        
        # 总能量
        total_energy = atomic_energies.sum()
        
        return total_energy
    
    def compute_forces(self, positions):
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        
        forces = -torch.autograd.grad(energy, positions)[0]
        return forces
```

---

## 7.2.5 分子动力学积分

### Velocity Verlet 积分器

```python
class VelocityVerlet:
    """
    Velocity Verlet 积分器
    
    r(t+dt) = r(t) + v(t)·dt + 0.5·a(t)·dt²
    v(t+dt) = v(t) + 0.5·[a(t) + a(t+dt)]·dt
    """
    
    def __init__(self, potential, dt=0.001, masses=None):
        """
        Args:
            potential: 势能函数模块
            dt: 时间步长
            masses: 原子质量 [N]
        """
        self.potential = potential
        self.dt = dt
        self.masses = masses
    
    def step(self, positions, velocities, forces=None):
        """
        执行一步积分
        
        Args:
            positions: [N, 3]
            velocities: [N, 3]
            forces: [N, 3]（可选，如果没有则计算）
        
        Returns:
            new_positions, new_velocities, new_forces
        """
        if forces is None:
            forces = self.potential.compute_forces(positions)
        
        if self.masses is None:
            masses = torch.ones(positions.shape[0], 1)
        else:
            masses = self.masses.reshape(-1, 1)
        
        # 计算加速度
        accelerations = forces / masses
        
        # 更新位置
        new_positions = positions + velocities * self.dt + 0.5 * accelerations * self.dt**2
        
        # 计算新的力
        new_forces = self.potential.compute_forces(new_positions)
        new_accelerations = new_forces / masses
        
        # 更新速度
        new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * self.dt
        
        return new_positions.detach(), new_velocities.detach(), new_forces.detach()


def run_md_simulation(potential, initial_positions, initial_velocities,
                       n_steps, dt=0.001):
    """
    运行 MD 模拟
    
    Args:
        potential: 势能函数
        initial_positions: [N, 3]
        initial_velocities: [N, 3]
        n_steps: 模拟步数
        dt: 时间步长
    
    Returns:
        trajectory: [n_steps, N, 3]
        energies: [n_steps]
    """
    integrator = VelocityVerlet(potential, dt)
    
    positions = initial_positions.clone()
    velocities = initial_velocities.clone()
    forces = potential.compute_forces(positions)
    
    trajectory = [positions.clone()]
    kinetic_energies = []
    potential_energies = []
    
    for step in range(n_steps):
        positions, velocities, forces = integrator.step(positions, velocities, forces)
        
        # 记录
        trajectory.append(positions.clone())
        
        # 计算能量
        KE = 0.5 * (velocities ** 2).sum()
        PE = potential(positions)
        
        kinetic_energies.append(KE.item())
        potential_energies.append(PE.item() if isinstance(PE, torch.Tensor) else PE)
    
    trajectory = torch.stack(trajectory)
    
    return trajectory, kinetic_energies, potential_energies
```

---

## 7.2.6 训练神经网络势函数

### 从 DFT 数据训练

```python
def train_nn_potential(model, train_data, val_data, n_epochs=1000):
    """
    从第一性原理数据训练神经网络势函数
    
    Args:
        model: 神经网络势函数
        train_data: [(positions, energy, forces), ...]
        val_data: 验证数据
        n_epochs: 训练轮数
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
    
    # 损失权重
    w_energy = 1.0
    w_forces = 10.0
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        for positions, energy_true, forces_true in train_data:
            optimizer.zero_grad()
            
            positions = positions.requires_grad_(True)
            
            # 预测能量
            energy_pred = model(positions)
            
            # 预测力
            forces_pred = -torch.autograd.grad(
                energy_pred, positions,
                create_graph=True
            )[0]
            
            # 损失
            loss_energy = (energy_pred - energy_true) ** 2
            loss_forces = ((forces_pred - forces_true) ** 2).mean()
            
            loss = w_energy * loss_energy + w_forces * loss_forces
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_data)
        history['train_loss'].append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for positions, energy_true, forces_true in val_data:
                positions = positions.requires_grad_(True)
                energy_pred = model(positions)
                forces_pred = model.compute_forces(positions)
                
                loss = (energy_pred - energy_true) ** 2
                loss += ((forces_pred - forces_true) ** 2).mean()
                val_loss += loss.item()
        
        val_loss /= len(val_data)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
    
    return history
```

---

## 7.2.7 完整示例：双原子分子

```python
def diatomic_molecule_example():
    """
    双原子分子振动模拟
    
    使用 Morse 势：V(r) = D·(1 - exp(-a(r-r0)))²
    """
    
    # Morse 势参数
    D = 1.0  # 解离能
    a = 1.0  # 势阱宽度
    r0 = 1.5  # 平衡键长
    
    class MorsePotential(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, positions):
            r = torch.norm(positions[1] - positions[0])
            return D * (1 - torch.exp(-a * (r - r0))) ** 2
        
        def compute_forces(self, positions):
            positions = positions.clone().requires_grad_(True)
            energy = self.forward(positions)
            forces = -torch.autograd.grad(energy, positions)[0]
            return forces
    
    potential = MorsePotential()
    
    # 初始条件：略微压缩的键
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0]  # r < r0，压缩
    ], dtype=torch.float32)
    
    initial_velocities = torch.zeros_like(initial_positions)
    
    # 运行模拟
    trajectory, KE, PE = run_md_simulation(
        potential, initial_positions, initial_velocities,
        n_steps=2000, dt=0.01
    )
    
    # 计算键长随时间的变化
    bond_lengths = []
    for pos in trajectory:
        r = torch.norm(pos[1] - pos[0])
        bond_lengths.append(r.item())
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 键长振动
    time = torch.arange(len(bond_lengths)) * 0.01
    axes[0, 0].plot(time.numpy(), bond_lengths)
    axes[0, 0].axhline(y=r0, color='r', linestyle='--', label=f'r₀={r0}')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('键长')
    axes[0, 0].set_title('键长振动')
    axes[0, 0].legend()
    
    # 能量
    total_energy = [k + p for k, p in zip(KE, PE)]
    axes[0, 1].plot(time[:-1].numpy(), KE, label='动能')
    axes[0, 1].plot(time[:-1].numpy(), PE, label='势能')
    axes[0, 1].plot(time[:-1].numpy(), total_energy, label='总能量')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('能量')
    axes[0, 1].set_title('能量守恒')
    axes[0, 1].legend()
    
    # 相空间
    v_bond = [(trajectory[i+1, 1, 0] - trajectory[i, 1, 0]).item() / 0.01 
              for i in range(len(trajectory)-1)]
    axes[1, 0].plot(bond_lengths[:-1], v_bond)
    axes[1, 0].set_xlabel('键长 r')
    axes[1, 0].set_ylabel('键长变化率 dr/dt')
    axes[1, 0].set_title('相空间轨迹')
    
    # Morse 势曲线
    r_range = torch.linspace(1.0, 3.0, 100)
    V_morse = D * (1 - torch.exp(-a * (r_range - r0))) ** 2
    axes[1, 1].plot(r_range.numpy(), V_morse.numpy())
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('V(r)')
    axes[1, 1].set_title('Morse 势能曲线')
    
    plt.tight_layout()
    plt.savefig('diatomic_md.png', dpi=150)
    plt.show()
    
    return trajectory

diatomic_molecule_example()
```

---

## 7.2.8 多体系统：Lennard-Jones 流体

```python
def lj_fluid_simulation():
    """
    Lennard-Jones 流体模拟
    """
    
    class LJPotential(nn.Module):
        def __init__(self, epsilon=1.0, sigma=1.0, r_cut=2.5):
            super().__init__()
            self.epsilon = epsilon
            self.sigma = sigma
            self.r_cut = r_cut * sigma
        
        def forward(self, positions):
            N = positions.shape[0]
            energy = torch.tensor(0.0)
            
            for i in range(N):
                for j in range(i + 1, N):
                    r = torch.norm(positions[i] - positions[j])
                    if r < self.r_cut:
                        sr6 = (self.sigma / r) ** 6
                        sr12 = sr6 ** 2
                        energy += 4 * self.epsilon * (sr12 - sr6)
            
            return energy
        
        def compute_forces(self, positions):
            positions = positions.clone().requires_grad_(True)
            energy = self.forward(positions)
            forces = -torch.autograd.grad(energy, positions)[0]
            return forces
    
    # 初始化：简单立方格子
    n_per_dim = 3
    spacing = 1.5
    
    positions = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                positions.append([i * spacing, j * spacing, k * spacing])
    
    positions = torch.tensor(positions, dtype=torch.float32)
    
    # 随机初始速度（Maxwell-Boltzmann 分布）
    temperature = 1.0
    velocities = torch.randn_like(positions) * torch.sqrt(torch.tensor(temperature))
    
    # 去除质心速度
    velocities -= velocities.mean(dim=0)
    
    potential = LJPotential()
    
    print(f"粒子数: {positions.shape[0]}")
    print(f"初始势能: {potential(positions).item():.4f}")
    
    # 运行模拟
    trajectory, KE, PE = run_md_simulation(
        potential, positions, velocities,
        n_steps=500, dt=0.005
    )
    
    return trajectory, KE, PE
```

---

## 🔬 物理视角总结

### 神经网络势函数的优势

| 方面 | 经验力场 | DFT | NN 势函数 |
|------|---------|-----|----------|
| 精度 | 低 | 高 | 接近 DFT |
| 速度 | 快 | 慢 | 快 |
| 可迁移性 | 有限 | 好 | 中等 |
| 系统规模 | 大 | 小 | 大 |

### 关键挑战

1. **对称性**：必须满足平移、旋转、置换不变性
2. **长程相互作用**：需要特殊处理
3. **外推**：在训练数据之外的构型可能不可靠
4. **多元素系统**：需要处理不同原子类型

---

## 📝 练习

1. 实现一个简单的 LJ 流体模拟，计算径向分布函数
2. 训练一个神经网络势函数拟合 Morse 势
3. 实现周期性边界条件

---

## ⏭️ 下一节

下一节我们将学习 [量子系统模拟](./03_quantum_systems.md)，了解如何用神经网络求解量子力学问题。

