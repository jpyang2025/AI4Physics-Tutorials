#!/usr/bin/env python3
"""
分子动力学模拟示例

本示例包含：
1. Lennard-Jones 势能
2. 神经网络势函数
3. Velocity Verlet 积分器
4. 双原子分子振动
5. 多粒子系统模拟

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. 势能函数
# =============================================================================

class LennardJonesPotential(nn.Module):
    """Lennard-Jones 势能"""
    
    def __init__(self, epsilon=1.0, sigma=1.0, r_cut=2.5):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.r_cut = r_cut * sigma
    
    def pair_energy(self, r):
        """计算一对粒子的势能"""
        if r > self.r_cut:
            return torch.tensor(0.0)
        
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2
        return 4 * self.epsilon * (sr12 - sr6)
    
    def forward(self, positions):
        """计算总势能"""
        N = positions.shape[0]
        energy = torch.tensor(0.0)
        
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = torch.norm(positions[i] - positions[j])
                if r_ij < self.r_cut:
                    sr6 = (self.sigma / r_ij) ** 6
                    sr12 = sr6 ** 2
                    energy = energy + 4 * self.epsilon * (sr12 - sr6)
        
        return energy
    
    def compute_forces(self, positions):
        """计算力 F = -∇E"""
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        forces = -torch.autograd.grad(energy, positions)[0]
        return forces


class MorsePotential(nn.Module):
    """Morse 势能（适用于双原子分子）"""
    
    def __init__(self, D=1.0, a=1.0, r0=1.5):
        super().__init__()
        self.D = D      # 解离能
        self.a = a      # 势阱宽度
        self.r0 = r0    # 平衡键长
    
    def forward(self, positions):
        """假设只有两个原子"""
        r = torch.norm(positions[1] - positions[0])
        return self.D * (1 - torch.exp(-self.a * (r - self.r0))) ** 2
    
    def compute_forces(self, positions):
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        forces = -torch.autograd.grad(energy, positions)[0]
        return forces


# =============================================================================
# 2. 神经网络势函数
# =============================================================================

class NNPotential(nn.Module):
    """神经网络势函数"""
    
    def __init__(self, n_atoms, hidden_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        
        # 使用距离矩阵作为输入（满足平移和旋转不变性）
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
        
        # 计算成对距离
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)
        
        # 提取上三角（不含对角线）
        triu_indices = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1)
        pair_distances = distances[:, triu_indices[0], triu_indices[1]]
        
        return self.net(pair_distances)
    
    def compute_forces(self, positions):
        positions = positions.clone().requires_grad_(True)
        energy = self.forward(positions)
        forces = -torch.autograd.grad(energy.sum(), positions)[0]
        return forces


# =============================================================================
# 3. 积分器
# =============================================================================

class VelocityVerletIntegrator:
    """Velocity Verlet 积分器"""
    
    def __init__(self, potential, dt=0.001, masses=None):
        self.potential = potential
        self.dt = dt
        self.masses = masses
    
    def step(self, positions, velocities, forces=None):
        """执行一步积分"""
        if forces is None:
            forces = self.potential.compute_forces(positions)
        
        if self.masses is None:
            masses = torch.ones(positions.shape[0], 1)
        else:
            masses = self.masses.reshape(-1, 1)
        
        # 加速度
        acc = forces / masses
        
        # 更新位置
        new_positions = positions + velocities * self.dt + 0.5 * acc * self.dt**2
        
        # 新的力和加速度
        new_forces = self.potential.compute_forces(new_positions)
        new_acc = new_forces / masses
        
        # 更新速度
        new_velocities = velocities + 0.5 * (acc + new_acc) * self.dt
        
        return new_positions.detach(), new_velocities.detach(), new_forces.detach()


def run_simulation(potential, positions, velocities, n_steps, dt=0.001):
    """运行 MD 模拟"""
    integrator = VelocityVerletIntegrator(potential, dt)
    
    trajectory = [positions.clone()]
    energies = {'kinetic': [], 'potential': [], 'total': []}
    
    forces = potential.compute_forces(positions)
    
    for _ in tqdm(range(n_steps), desc="MD Simulation"):
        positions, velocities, forces = integrator.step(positions, velocities, forces)
        
        trajectory.append(positions.clone())
        
        KE = 0.5 * (velocities ** 2).sum().item()
        PE = potential(positions).item()
        
        energies['kinetic'].append(KE)
        energies['potential'].append(PE)
        energies['total'].append(KE + PE)
    
    return torch.stack(trajectory), energies


# =============================================================================
# 4. 示例：双原子分子
# =============================================================================

def diatomic_example():
    """双原子分子振动模拟"""
    print("\n" + "=" * 60)
    print("示例 1: 双原子分子振动")
    print("=" * 60)
    
    potential = MorsePotential(D=1.0, a=1.0, r0=1.5)
    
    # 初始位置（略微压缩的键）
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0]  # r < r0
    ], dtype=torch.float32)
    
    velocities = torch.zeros_like(positions)
    
    # 运行模拟
    trajectory, energies = run_simulation(
        potential, positions, velocities, 
        n_steps=2000, dt=0.01
    )
    
    # 计算键长
    bond_lengths = [torch.norm(pos[1] - pos[0]).item() for pos in trajectory]
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time = np.arange(len(bond_lengths)) * 0.01
    
    # 键长振动
    axes[0, 0].plot(time, bond_lengths)
    axes[0, 0].axhline(y=potential.r0, color='r', linestyle='--', label=f'r₀ = {potential.r0}')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('键长')
    axes[0, 0].set_title('键长振动')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 能量
    axes[0, 1].plot(time[:-1], energies['kinetic'], label='动能', alpha=0.8)
    axes[0, 1].plot(time[:-1], energies['potential'], label='势能', alpha=0.8)
    axes[0, 1].plot(time[:-1], energies['total'], label='总能量', linewidth=2)
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('能量')
    axes[0, 1].set_title('能量守恒')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 能量守恒检验
    energy_drift = np.array(energies['total']) - energies['total'][0]
    axes[1, 0].plot(time[:-1], energy_drift)
    axes[1, 0].set_xlabel('时间')
    axes[1, 0].set_ylabel('能量漂移')
    axes[1, 0].set_title('能量守恒检验')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 相空间
    dr_dt = np.diff(bond_lengths) / 0.01
    axes[1, 1].plot(bond_lengths[:-1], dr_dt)
    axes[1, 1].set_xlabel('键长 r')
    axes[1, 1].set_ylabel('dr/dt')
    axes[1, 1].set_title('相空间轨迹')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('md_diatomic.png', dpi=150)
    plt.show()
    
    return trajectory


# =============================================================================
# 5. 示例：多粒子系统
# =============================================================================

def multiparticle_example():
    """多粒子 LJ 系统模拟"""
    print("\n" + "=" * 60)
    print("示例 2: 多粒子 Lennard-Jones 系统")
    print("=" * 60)
    
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, r_cut=2.5)
    
    # 初始化：简单立方格子
    n_per_dim = 3
    spacing = 1.5
    
    positions = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                positions.append([i * spacing, j * spacing, k * spacing])
    
    positions = torch.tensor(positions, dtype=torch.float32)
    n_atoms = positions.shape[0]
    
    print(f"粒子数: {n_atoms}")
    print(f"初始势能: {potential(positions).item():.4f}")
    
    # 随机初始速度
    temperature = 0.5
    velocities = torch.randn_like(positions) * np.sqrt(temperature)
    velocities -= velocities.mean(dim=0)  # 去除质心速度
    
    # 运行模拟
    trajectory, energies = run_simulation(
        potential, positions, velocities,
        n_steps=500, dt=0.005
    )
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time = np.arange(len(energies['total'])) * 0.005
    
    # 能量
    axes[0, 0].plot(time, energies['kinetic'], label='动能', alpha=0.8)
    axes[0, 0].plot(time, energies['potential'], label='势能', alpha=0.8)
    axes[0, 0].plot(time, energies['total'], label='总能量', linewidth=2)
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('能量')
    axes[0, 0].set_title('系统能量')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 温度（从动能估算）
    # KE = (3/2) N kT, 这里 k=1
    temperatures = [2 * ke / (3 * n_atoms) for ke in energies['kinetic']]
    axes[0, 1].plot(time, temperatures)
    axes[0, 1].axhline(y=temperature, color='r', linestyle='--', label=f'初始 T = {temperature}')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('温度')
    axes[0, 1].set_title('系统温度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 初始和最终构型
    pos_init = trajectory[0].numpy()
    pos_final = trajectory[-1].numpy()
    
    ax1 = fig.add_subplot(2, 2, 3, projection='3d')
    ax1.scatter(pos_init[:, 0], pos_init[:, 1], pos_init[:, 2], s=100)
    ax1.set_title('初始构型')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    ax2.scatter(pos_final[:, 0], pos_final[:, 1], pos_final[:, 2], s=100)
    ax2.set_title('最终构型')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    plt.tight_layout()
    plt.savefig('md_multiparticle.png', dpi=150)
    plt.show()
    
    return trajectory


# =============================================================================
# 6. 训练神经网络势函数
# =============================================================================

def train_nn_potential():
    """训练神经网络拟合 LJ 势"""
    print("\n" + "=" * 60)
    print("示例 3: 训练神经网络势函数")
    print("=" * 60)
    
    n_atoms = 3
    
    # 参考势函数
    ref_potential = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    
    # 神经网络势函数
    nn_potential = NNPotential(n_atoms, hidden_dim=64)
    optimizer = optim.Adam(nn_potential.parameters(), lr=0.001)
    
    # 生成训练数据
    n_samples = 1000
    
    train_data = []
    for _ in range(n_samples):
        # 随机构型
        positions = torch.randn(n_atoms, 3) * 2
        positions = positions.requires_grad_(True)
        
        # 参考能量和力
        energy_ref = ref_potential(positions)
        forces_ref = ref_potential.compute_forces(positions)
        
        train_data.append((positions.detach(), energy_ref.detach(), forces_ref.detach()))
    
    # 训练
    losses = []
    
    for epoch in tqdm(range(500), desc="Training NN Potential"):
        epoch_loss = 0
        
        for positions, energy_ref, forces_ref in train_data:
            optimizer.zero_grad()
            
            positions = positions.requires_grad_(True)
            
            energy_pred = nn_potential(positions)
            forces_pred = -torch.autograd.grad(
                energy_pred.sum(), positions, create_graph=True
            )[0]
            
            loss_energy = (energy_pred - energy_ref) ** 2
            loss_forces = ((forces_pred - forces_ref) ** 2).mean()
            
            loss = loss_energy + 10 * loss_forces
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_data))
    
    # 测试
    test_positions = torch.randn(n_atoms, 3) * 2
    
    with torch.no_grad():
        energy_ref = ref_potential(test_positions)
        energy_pred = nn_potential(test_positions)
    
    print(f"\n测试能量:")
    print(f"  参考: {energy_ref.item():.4f}")
    print(f"  预测: {energy_pred.item():.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)
    
    # 比较双体势
    r_range = torch.linspace(0.9, 3.0, 50)
    
    ref_energies = []
    nn_energies = []
    
    for r in r_range:
        pos = torch.tensor([[0, 0, 0], [r.item(), 0, 0], [5, 5, 5]], dtype=torch.float32)
        
        with torch.no_grad():
            ref_energies.append(ref_potential(pos).item())
            nn_energies.append(nn_potential(pos).item())
    
    plt.subplot(1, 2, 2)
    plt.plot(r_range.numpy(), ref_energies, 'b-', label='LJ 参考', linewidth=2)
    plt.plot(r_range.numpy(), nn_energies, 'r--', label='NN 预测', linewidth=2)
    plt.xlabel('r')
    plt.ylabel('能量')
    plt.title('势能曲线比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('md_nn_potential.png', dpi=150)
    plt.show()
    
    return nn_potential


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("分子动力学模拟示例")
    print("=" * 60)
    
    diatomic_example()
    multiparticle_example()
    train_nn_potential()
    
    print("\n所有示例完成！")


def demo():
    """快速演示"""
    print("分子动力学快速演示")
    print("=" * 60)
    
    potential = MorsePotential(D=1.0, a=1.0, r0=1.5)
    
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    velocities = torch.zeros_like(positions)
    
    trajectory, energies = run_simulation(
        potential, positions, velocities,
        n_steps=500, dt=0.01
    )
    
    bond_lengths = [torch.norm(pos[1] - pos[0]).item() for pos in trajectory]
    
    print(f"初始键长: {bond_lengths[0]:.3f}")
    print(f"最大键长: {max(bond_lengths):.3f}")
    print(f"最小键长: {min(bond_lengths):.3f}")
    print(f"平衡键长: {potential.r0}")
    print(f"能量守恒: 最大漂移 = {max(energies['total']) - min(energies['total']):.6f}")
    
    print("\n快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        main()

