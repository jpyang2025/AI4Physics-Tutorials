#!/usr/bin/env python3
"""
量子力学模拟示例

本示例包含：
1. 一维无限深势阱
2. 量子谐振子
3. 氢原子（径向波函数）
4. 激发态求解
5. 变分蒙特卡洛 (VMC)

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. 波函数神经网络
# =============================================================================

class WavefunctionNN(nn.Module):
    """通用波函数神经网络"""
    
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=3):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class BoundaryConditionNN(nn.Module):
    """自动满足边界条件的神经网络"""
    
    def __init__(self, L=1.0, hidden_dim=32):
        super().__init__()
        self.L = L
        self.net = WavefunctionNN(hidden_dim=hidden_dim)
    
    def forward(self, x):
        # ψ(0) = ψ(L) = 0
        return x * (self.L - x) * self.net(x)


class GaussianEnvelopeNN(nn.Module):
    """带高斯包络的神经网络（适用于无界问题）"""
    
    def __init__(self, alpha=0.5, hidden_dim=32):
        super().__init__()
        self.alpha = alpha
        self.net = WavefunctionNN(hidden_dim=hidden_dim)
    
    def forward(self, x):
        envelope = torch.exp(-self.alpha * x ** 2)
        return envelope * self.net(x)


# =============================================================================
# 2. 辅助函数
# =============================================================================

def compute_derivatives(model, x):
    """计算波函数的一阶和二阶导数"""
    x = x.requires_grad_(True)
    psi = model(x)
    
    dpsi = torch.autograd.grad(
        psi, x, grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]
    
    d2psi = torch.autograd.grad(
        dpsi, x, grad_outputs=torch.ones_like(dpsi),
        create_graph=True
    )[0]
    
    return psi, dpsi, d2psi


def normalize(psi, dx):
    """归一化波函数"""
    norm = (psi ** 2).sum() * dx
    return psi / torch.sqrt(norm)


# =============================================================================
# 3. 无限深势阱
# =============================================================================

def solve_infinite_well():
    """
    求解一维无限深势阱
    
    -ℏ²/(2m) d²ψ/dx² = Eψ, 0 < x < L
    ψ(0) = ψ(L) = 0
    
    解析解: ψ_n = √(2/L) sin(nπx/L), E_n = n²π²ℏ²/(2mL²)
    """
    print("\n" + "=" * 60)
    print("示例 1: 一维无限深势阱")
    print("=" * 60)
    
    L = 1.0
    hbar = 1.0
    m = 1.0
    
    model = BoundaryConditionNN(L=L, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.linspace(0.001, L - 0.001, 100, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in tqdm(range(3000), desc="Training"):
        optimizer.zero_grad()
        
        psi, dpsi, d2psi = compute_derivatives(model, x)
        
        # 归一化
        psi_n = normalize(psi, dx)
        d2psi_n = d2psi / torch.sqrt((psi ** 2).sum() * dx)
        
        # 动能期望值
        kinetic = -hbar**2 / (2 * m) * (psi_n * d2psi_n).sum() * dx
        
        energy = kinetic
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
    
    # 解析解
    E_exact = np.pi**2 * hbar**2 / (2 * m * L**2)
    print(f"\n计算能量: {energy_history[-1]:.6f}")
    print(f"精确能量: {E_exact:.6f}")
    print(f"相对误差: {abs(energy_history[-1] - E_exact) / E_exact * 100:.2f}%")
    
    # 可视化
    model.eval()
    x_plot = torch.linspace(0, L, 100).reshape(-1, 1)
    
    with torch.no_grad():
        psi_pred = model(x_plot)
        psi_pred = normalize(psi_pred, L / 100)
    
    psi_exact = np.sqrt(2 / L) * np.sin(np.pi * x_plot.numpy() / L)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(x_plot.numpy(), psi_pred.numpy(), 'r-', label='NN', linewidth=2)
    axes[0].plot(x_plot.numpy(), psi_exact, 'b--', label='解析解', linewidth=2)
    axes[0].plot(x_plot.numpy(), -psi_exact, 'b--', alpha=0.3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ψ(x)')
    axes[0].set_title('基态波函数')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(energy_history)
    axes[1].axhline(y=E_exact, color='r', linestyle='--', label=f'E = {E_exact:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('能量收敛')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qm_infinite_well.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 4. 量子谐振子
# =============================================================================

def solve_harmonic_oscillator():
    """
    求解量子谐振子
    
    H = -ℏ²/(2m) d²/dx² + (1/2)mω²x²
    
    基态: ψ_0 = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ), E_0 = ℏω/2
    """
    print("\n" + "=" * 60)
    print("示例 2: 量子谐振子")
    print("=" * 60)
    
    # 使用自然单位 ℏ = m = ω = 1
    hbar = m = omega = 1.0
    
    model = GaussianEnvelopeNN(alpha=0.5, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.linspace(-5, 5, 200, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in tqdm(range(5000), desc="Training"):
        optimizer.zero_grad()
        
        psi, dpsi, d2psi = compute_derivatives(model, x)
        
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
    
    E_exact = 0.5 * hbar * omega
    print(f"\n计算能量: {energy_history[-1]:.6f}")
    print(f"精确能量: {E_exact:.6f}")
    
    # 可视化
    model.eval()
    x_plot = torch.linspace(-5, 5, 200).reshape(-1, 1)
    
    with torch.no_grad():
        psi_pred = model(x_plot)
        psi_pred = normalize(psi_pred, 0.05)
    
    psi_exact = (1 / np.pi) ** 0.25 * np.exp(-0.5 * x_plot.numpy() ** 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x_plot.numpy(), psi_pred.numpy(), 'r-', label='NN', linewidth=2)
    axes[0].plot(x_plot.numpy(), psi_exact, 'b--', label='解析解', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ψ(x)')
    axes[0].set_title('基态波函数')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_plot.numpy(), psi_pred.numpy()**2, 'r-', label='NN', linewidth=2)
    axes[1].plot(x_plot.numpy(), psi_exact**2, 'b--', label='解析解', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|ψ(x)|²')
    axes[1].set_title('概率密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(energy_history)
    axes[2].axhline(y=E_exact, color='r', linestyle='--', label=f'E₀ = {E_exact}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('能量收敛')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qm_harmonic_oscillator.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 5. 氢原子
# =============================================================================

class HydrogenNN(nn.Module):
    """氢原子径向波函数"""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = WavefunctionNN(hidden_dim=hidden_dim)
    
    def forward(self, r):
        # 确保正确渐近行为: r→∞ 指数衰减
        envelope = torch.exp(-r)
        return envelope * self.net(r)


def solve_hydrogen_atom():
    """
    求解氢原子基态（原子单位）
    
    H = -1/2 (d²/dr² + 2/r d/dr) - 1/r
    
    基态: R_1s = 2 exp(-r), E = -0.5 Hartree
    """
    print("\n" + "=" * 60)
    print("示例 3: 氢原子基态")
    print("=" * 60)
    
    model = HydrogenNN(hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    r = torch.linspace(0.01, 15, 200, requires_grad=True).reshape(-1, 1)
    dr = r[1, 0] - r[0, 0]
    
    energy_history = []
    
    for epoch in tqdm(range(5000), desc="Training"):
        optimizer.zero_grad()
        
        R = model(r)
        
        dR = torch.autograd.grad(R, r, torch.ones_like(R), create_graph=True)[0]
        d2R = torch.autograd.grad(dR, r, torch.ones_like(dR), create_graph=True)[0]
        
        # 归一化（径向: ∫|R|² r² dr = 1）
        norm = (R**2 * r**2).sum() * dr
        R_n = R / torch.sqrt(norm)
        dR_n = dR / torch.sqrt(norm)
        d2R_n = d2R / torch.sqrt(norm)
        
        # 动能
        T = -0.5 * ((d2R_n + 2/r * dR_n) * R_n * r**2).sum() * dr
        
        # 势能
        V = -(R_n**2 * r).sum() * dr
        
        energy = T + V
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
    
    E_exact = -0.5
    print(f"\n计算能量: {energy_history[-1]:.6f} Hartree")
    print(f"精确能量: {E_exact} Hartree")
    
    # 可视化
    model.eval()
    r_plot = torch.linspace(0.01, 10, 200).reshape(-1, 1)
    
    with torch.no_grad():
        R_pred = model(r_plot)
        norm = (R_pred**2 * r_plot**2).sum() * 0.05
        R_pred = R_pred / torch.sqrt(norm)
    
    R_exact = 2 * np.exp(-r_plot.numpy())
    R_exact = R_exact / np.sqrt((R_exact**2 * r_plot.numpy()**2).sum() * 0.05)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(r_plot.numpy(), R_pred.numpy(), 'r-', label='NN', linewidth=2)
    axes[0].plot(r_plot.numpy(), R_exact, 'b--', label='解析解', linewidth=2)
    axes[0].set_xlabel('r (a₀)')
    axes[0].set_ylabel('R(r)')
    axes[0].set_title('径向波函数')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    P_pred = R_pred.numpy()**2 * r_plot.numpy()**2
    P_exact = R_exact**2 * r_plot.numpy()**2
    
    axes[1].plot(r_plot.numpy(), P_pred, 'r-', label='NN', linewidth=2)
    axes[1].plot(r_plot.numpy(), P_exact, 'b--', label='解析解', linewidth=2)
    axes[1].set_xlabel('r (a₀)')
    axes[1].set_ylabel('r²|R(r)|²')
    axes[1].set_title('径向概率密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(energy_history)
    axes[2].axhline(y=E_exact, color='r', linestyle='--', label='E = -0.5 Hartree')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Energy (Hartree)')
    axes[2].set_title('能量收敛')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qm_hydrogen.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 6. 激发态
# =============================================================================

class ExcitedStateNN(nn.Module):
    """激发态波函数（与基态正交）"""
    
    def __init__(self, ground_state_model, hidden_dim=32):
        super().__init__()
        self.ground_state = ground_state_model
        for param in self.ground_state.parameters():
            param.requires_grad = False
        
        self.net = GaussianEnvelopeNN(alpha=0.5, hidden_dim=hidden_dim)
    
    def forward(self, x, dx):
        psi = self.net(x)
        
        with torch.no_grad():
            psi_ground = self.ground_state(x)
            psi_ground = normalize(psi_ground, dx)
        
        # Gram-Schmidt 正交化
        overlap = (psi * psi_ground).sum() * dx
        psi_orth = psi - overlap * psi_ground
        
        return psi_orth


def solve_excited_state(ground_model):
    """求解第一激发态"""
    print("\n" + "=" * 60)
    print("示例 4: 谐振子第一激发态")
    print("=" * 60)
    
    model = ExcitedStateNN(ground_model, hidden_dim=32)
    optimizer = optim.Adam(model.net.parameters(), lr=0.01)
    
    x = torch.linspace(-6, 6, 200, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in tqdm(range(5000), desc="Training"):
        optimizer.zero_grad()
        
        psi = model(x, dx)
        
        # 归一化
        norm = (psi ** 2).sum() * dx
        psi_n = psi / torch.sqrt(norm)
        
        # 导数
        dpsi = torch.autograd.grad(psi_n.sum(), x, create_graph=True)[0]
        d2psi = torch.autograd.grad(dpsi.sum(), x, create_graph=True)[0]
        
        # 能量
        T = -0.5 * (psi_n * d2psi).sum() * dx
        V = 0.5 * (x**2 * psi_n**2).sum() * dx
        
        energy = T + V
        
        # 正交性惩罚
        with torch.no_grad():
            psi_ground = ground_model(x)
            psi_ground = normalize(psi_ground, dx)
        
        overlap = (psi_n * psi_ground).sum() * dx
        penalty = 100 * overlap ** 2
        
        loss = energy + penalty
        
        loss.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
    
    E_exact = 1.5  # ℏω(n + 1/2), n=1
    print(f"\n计算能量: {energy_history[-1]:.6f}")
    print(f"精确能量: {E_exact}")
    
    # 可视化
    model.eval()
    ground_model.eval()
    
    x_plot = torch.linspace(-6, 6, 200).reshape(-1, 1)
    
    with torch.no_grad():
        psi0 = ground_model(x_plot)
        psi0 = normalize(psi0, 0.06)
        
        psi1 = model(x_plot, torch.tensor(0.06))
        psi1 = normalize(psi1, 0.06)
    
    # 解析解
    psi0_exact = (1/np.pi)**0.25 * np.exp(-0.5 * x_plot.numpy()**2)
    psi1_exact = (4/np.pi)**0.25 * x_plot.numpy() * np.exp(-0.5 * x_plot.numpy()**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(x_plot.numpy(), psi0.numpy(), 'b-', label='n=0 (NN)', linewidth=2)
    axes[0].plot(x_plot.numpy(), psi1.numpy(), 'r-', label='n=1 (NN)', linewidth=2)
    axes[0].plot(x_plot.numpy(), psi0_exact, 'b--', alpha=0.5, label='n=0 (精确)')
    axes[0].plot(x_plot.numpy(), psi1_exact / np.sqrt((psi1_exact**2).sum() * 0.06), 
                  'r--', alpha=0.5, label='n=1 (精确)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ψ(x)')
    axes[0].set_title('基态和第一激发态')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(energy_history)
    axes[1].axhline(y=E_exact, color='r', linestyle='--', label=f'E₁ = {E_exact}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('激发态能量收敛')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qm_excited_state.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 7. 双势阱
# =============================================================================

def solve_double_well():
    """
    求解双势阱问题
    
    V(x) = (x² - 1)² = x⁴ - 2x² + 1
    """
    print("\n" + "=" * 60)
    print("示例 5: 双势阱")
    print("=" * 60)
    
    model = GaussianEnvelopeNN(alpha=0.3, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    x = torch.linspace(-3, 3, 200, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    energy_history = []
    
    for epoch in tqdm(range(8000), desc="Training"):
        optimizer.zero_grad()
        
        psi, dpsi, d2psi = compute_derivatives(model, x)
        
        norm = (psi ** 2).sum() * dx
        psi_n = psi / torch.sqrt(norm)
        d2psi_n = d2psi / torch.sqrt(norm)
        
        # 动能
        T = -0.5 * (psi_n * d2psi_n).sum() * dx
        
        # 势能 V = x⁴ - 2x² + 1
        V_potential = x**4 - 2*x**2 + 1
        V = (V_potential * psi_n**2).sum() * dx
        
        energy = T + V
        
        energy.backward()
        optimizer.step()
        
        energy_history.append(energy.item())
    
    print(f"\n基态能量: {energy_history[-1]:.6f}")
    
    # 可视化
    model.eval()
    x_plot = torch.linspace(-3, 3, 200).reshape(-1, 1)
    
    with torch.no_grad():
        psi_pred = model(x_plot)
        psi_pred = normalize(psi_pred, 0.03)
    
    V_plot = (x_plot**4 - 2*x_plot**2 + 1).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax2 = ax1.twinx()
    
    ax1.plot(x_plot.numpy(), V_plot, 'b-', label='V(x)', linewidth=2)
    ax2.plot(x_plot.numpy(), psi_pred.numpy()**2 * 5, 'r-', label='|ψ|²', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('V(x)', color='b')
    ax2.set_ylabel('|ψ(x)|²', color='r')
    ax1.set_title('双势阱中的基态')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    axes[1].plot(energy_history)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('能量收敛')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qm_double_well.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有量子力学示例"""
    print("量子力学模拟示例")
    print("=" * 60)
    
    solve_infinite_well()
    ground_model = solve_harmonic_oscillator()
    solve_hydrogen_atom()
    solve_excited_state(ground_model)
    solve_double_well()
    
    print("\n所有示例完成！")


def demo():
    """快速演示"""
    print("量子力学快速演示 - 无限深势阱")
    print("=" * 60)
    
    L = 1.0
    model = BoundaryConditionNN(L=L, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.linspace(0.01, L - 0.01, 50, requires_grad=True).reshape(-1, 1)
    dx = x[1, 0] - x[0, 0]
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        psi, _, d2psi = compute_derivatives(model, x)
        psi_n = normalize(psi, dx)
        d2psi_n = d2psi / torch.sqrt((psi ** 2).sum() * dx)
        
        energy = -0.5 * (psi_n * d2psi_n).sum() * dx
        
        energy.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: E = {energy.item():.6f}")
    
    E_exact = np.pi**2 / 2
    print(f"\n最终能量: {energy.item():.6f}")
    print(f"精确能量: {E_exact:.6f}")
    print(f"相对误差: {abs(energy.item() - E_exact) / E_exact * 100:.2f}%")
    
    print("\n快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        main()

