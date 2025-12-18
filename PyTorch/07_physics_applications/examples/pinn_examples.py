#!/usr/bin/env python3
"""
物理信息神经网络 (PINN) 示例

本示例包含：
1. 一阶 ODE 求解
2. 二阶 ODE（简谐振子）
3. 热传导方程
4. 波动方程
5. Burgers 方程
6. 逆问题（参数估计）

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. 基础 PINN 网络
# =============================================================================

class PINN(nn.Module):
    """通用 PINN 网络"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.net(x)


# =============================================================================
# 2. 一阶 ODE: dy/dx = -y, y(0) = 1
# =============================================================================

def solve_first_order_ode():
    """
    求解一阶 ODE: dy/dx = -y
    边界条件: y(0) = 1
    解析解: y = exp(-x)
    """
    print("\n" + "=" * 60)
    print("示例 1: 一阶 ODE (dy/dx = -y)")
    print("=" * 60)
    
    model = PINN(input_dim=1, output_dim=1, hidden_dims=[32, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 内部配点
    x_interior = torch.linspace(0, 5, 100, requires_grad=True).reshape(-1, 1)
    
    # 边界点
    x_bc = torch.zeros(1, 1)
    y_bc = torch.ones(1, 1)
    
    losses = []
    
    for epoch in tqdm(range(3000), desc="Training"):
        optimizer.zero_grad()
        
        # ODE 残差
        y = model(x_interior)
        dy_dx = torch.autograd.grad(
            y, x_interior, 
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        residual = dy_dx + y
        loss_ode = (residual ** 2).mean()
        
        # 边界条件
        y_pred_bc = model(x_bc)
        loss_bc = (y_pred_bc - y_bc) ** 2
        
        loss = loss_ode + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # 验证
    model.eval()
    x_test = torch.linspace(0, 5, 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test)
    y_exact = torch.exp(-x_test)
    
    error = (y_pred - y_exact).abs().max().item()
    print(f"最大误差: {error:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x_test.numpy(), y_pred.numpy(), 'r-', label='PINN', linewidth=2)
    axes[0].plot(x_test.numpy(), y_exact.numpy(), 'b--', label='解析解', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('ODE 解')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot((y_pred - y_exact).abs().numpy())
    axes[1].set_xlabel('x index')
    axes[1].set_ylabel('|误差|')
    axes[1].set_title('绝对误差')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('训练损失')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_first_order_ode.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 3. 二阶 ODE: 简谐振子 y'' + ω²y = 0
# =============================================================================

def solve_harmonic_oscillator(omega=2.0):
    """
    求解简谐振子: y'' + ω²y = 0
    初始条件: y(0) = 1, y'(0) = 0
    解析解: y = cos(ωx)
    """
    print("\n" + "=" * 60)
    print(f"示例 2: 简谐振子 (ω = {omega})")
    print("=" * 60)
    
    model = PINN(input_dim=1, output_dim=1, hidden_dims=[64, 64, 64])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    x_interior = torch.linspace(0.01, 2*np.pi, 200, requires_grad=True).reshape(-1, 1)
    x_ic = torch.zeros(1, 1, requires_grad=True)
    
    losses = []
    
    for epoch in tqdm(range(5000), desc="Training"):
        optimizer.zero_grad()
        
        # ODE 残差
        y = model(x_interior)
        dy = torch.autograd.grad(y, x_interior, torch.ones_like(y), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, x_interior, torch.ones_like(dy), create_graph=True)[0]
        
        residual = d2y + omega**2 * y
        loss_ode = (residual ** 2).mean()
        
        # 初始条件
        y_ic = model(x_ic)
        dy_ic = torch.autograd.grad(y_ic, x_ic, torch.ones_like(y_ic), create_graph=True)[0]
        
        loss_ic = (y_ic - 1.0)**2 + dy_ic**2
        
        loss = loss_ode + 100 * loss_ic
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # 验证
    model.eval()
    x_test = torch.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test)
    y_exact = torch.cos(omega * x_test)
    
    print(f"最大误差: {(y_pred - y_exact).abs().max().item():.6f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='PINN', linewidth=2)
    plt.plot(x_test.numpy(), y_exact.numpy(), 'b--', label='解析解', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('简谐振子解')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_harmonic_oscillator.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 4. 热传导方程
# =============================================================================

def solve_heat_equation(alpha=0.1, T_max=0.5):
    """
    求解热传导方程: ∂u/∂t = α ∂²u/∂x²
    初始条件: u(x, 0) = sin(πx)
    边界条件: u(0, t) = u(1, t) = 0
    解析解: u(x, t) = sin(πx) exp(-απ²t)
    """
    print("\n" + "=" * 60)
    print("示例 3: 热传导方程")
    print("=" * 60)
    
    model = PINN(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_interior = 2000
    n_bc = 100
    n_ic = 100
    
    losses = {'pde': [], 'ic': [], 'bc': [], 'total': []}
    
    for epoch in tqdm(range(10000), desc="Training"):
        optimizer.zero_grad()
        
        # 内部配点
        x_int = torch.rand(n_interior, 1, requires_grad=True)
        t_int = torch.rand(n_interior, 1, requires_grad=True) * T_max
        
        u = model(x_int, t_int)
        
        u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_t - alpha * u_xx
        loss_pde = (residual ** 2).mean()
        
        # 初始条件
        x_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)
        u_ic = model(x_ic, t_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = ((u_ic - u_ic_true) ** 2).mean()
        
        # 边界条件
        t_bc = torch.rand(n_bc, 1) * T_max
        u_left = model(torch.zeros(n_bc, 1), t_bc)
        u_right = model(torch.ones(n_bc, 1), t_bc)
        loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()
        
        loss = loss_pde + 10 * loss_ic + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        losses['pde'].append(loss_pde.item())
        losses['ic'].append(loss_ic.item())
        losses['bc'].append(loss_bc.item())
        losses['total'].append(loss.item())
    
    # 可视化
    model.eval()
    
    x_grid = torch.linspace(0, 1, 50)
    t_grid = torch.linspace(0, T_max, 50)
    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
    
    x_test = X.reshape(-1, 1)
    t_test = T.reshape(-1, 1)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test).reshape(50, 50)
    
    u_exact = torch.sin(np.pi * X) * torch.exp(-alpha * np.pi**2 * T)
    
    error = (u_pred - u_exact).abs()
    print(f"最大误差: {error.max().item():.6f}")
    print(f"平均误差: {error.mean().item():.6f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im1 = axes[0, 0].pcolormesh(X.numpy(), T.numpy(), u_pred.numpy(), 
                                  shading='auto', cmap='hot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('PINN 预测')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].pcolormesh(X.numpy(), T.numpy(), u_exact.numpy(), 
                                  shading='auto', cmap='hot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('解析解')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].pcolormesh(X.numpy(), T.numpy(), error.numpy(), 
                                  shading='auto', cmap='Blues')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title('绝对误差')
    plt.colorbar(im3, ax=axes[1, 0])
    
    axes[1, 1].semilogy(losses['pde'], label='PDE')
    axes[1, 1].semilogy(losses['ic'], label='IC')
    axes[1, 1].semilogy(losses['bc'], label='BC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('训练损失')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_heat_equation.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 5. 波动方程
# =============================================================================

def solve_wave_equation(c=1.0, T_max=2.0):
    """
    求解波动方程: ∂²u/∂t² = c² ∂²u/∂x²
    初始条件: u(x, 0) = sin(πx), ∂u/∂t(x, 0) = 0
    边界条件: u(0, t) = u(1, t) = 0
    解析解: u(x, t) = sin(πx) cos(cπt)
    """
    print("\n" + "=" * 60)
    print("示例 4: 波动方程")
    print("=" * 60)
    
    model = PINN(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in tqdm(range(15000), desc="Training"):
        optimizer.zero_grad()
        
        # 内部配点
        x_int = torch.rand(1000, 1, requires_grad=True)
        t_int = torch.rand(1000, 1, requires_grad=True) * T_max
        
        u = model(x_int, t_int)
        
        u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_int, torch.ones_like(u_t), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_tt - c**2 * u_xx
        loss_pde = (residual ** 2).mean()
        
        # 初始条件
        x_ic = torch.rand(100, 1, requires_grad=True)
        t_ic = torch.zeros(100, 1, requires_grad=True)
        
        u_ic = model(x_ic, t_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic1 = ((u_ic - u_ic_true) ** 2).mean()
        
        u_t_ic = torch.autograd.grad(u_ic.sum(), t_ic, create_graph=True)[0]
        loss_ic2 = (u_t_ic ** 2).mean()
        
        # 边界条件
        t_bc = torch.rand(100, 1) * T_max
        u_left = model(torch.zeros(100, 1), t_bc)
        u_right = model(torch.ones(100, 1), t_bc)
        loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()
        
        loss = loss_pde + 10 * (loss_ic1 + loss_ic2) + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
    
    # 可视化
    model.eval()
    
    x_grid = torch.linspace(0, 1, 50)
    t_grid = torch.linspace(0, T_max, 50)
    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.reshape(-1, 1), T.reshape(-1, 1)).reshape(50, 50)
    
    u_exact = torch.sin(np.pi * X) * torch.cos(c * np.pi * T)
    
    print(f"最大误差: {(u_pred - u_exact).abs().max().item():.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(X.numpy(), T.numpy(), u_pred.numpy(), 
                              shading='auto', cmap='RdBu')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('PINN 预测')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(X.numpy(), T.numpy(), u_exact.numpy(), 
                              shading='auto', cmap='RdBu')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('解析解')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_wave_equation.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 6. 逆问题：参数估计
# =============================================================================

class PINNInverse(nn.Module):
    """带可学习参数的 PINN"""
    
    def __init__(self):
        super().__init__()
        self.net = PINN(input_dim=2, output_dim=1, hidden_dims=[64, 64])
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 未知参数
    
    def forward(self, x, t):
        return self.net(x, t)


def solve_inverse_problem():
    """
    逆问题：从观测数据推断热扩散系数 α
    真实值 α = 0.1
    """
    print("\n" + "=" * 60)
    print("示例 5: 逆问题 - 参数估计")
    print("=" * 60)
    
    alpha_true = 0.1
    
    # 生成"观测"数据
    def exact_solution(x, t):
        return torch.sin(np.pi * x) * torch.exp(-alpha_true * np.pi**2 * t)
    
    n_obs = 50
    x_obs = torch.rand(n_obs, 1)
    t_obs = torch.rand(n_obs, 1) * 0.5
    u_obs = exact_solution(x_obs, t_obs) + 0.01 * torch.randn(n_obs, 1)
    
    model = PINNInverse()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    alpha_history = []
    
    for epoch in tqdm(range(10000), desc="Training"):
        optimizer.zero_grad()
        
        # 数据拟合
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
        
        with torch.no_grad():
            model.alpha.clamp_(min=0.001)
        
        alpha_history.append(model.alpha.item())
    
    print(f"\n估计值: α = {model.alpha.item():.4f}")
    print(f"真实值: α = {alpha_true}")
    print(f"相对误差: {abs(model.alpha.item() - alpha_true) / alpha_true * 100:.2f}%")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(alpha_history)
    plt.axhline(y=alpha_true, color='r', linestyle='--', label=f'真实值 = {alpha_true}')
    plt.xlabel('Epoch')
    plt.ylabel('α')
    plt.title('参数收敛')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 最终预测 vs 观测
    with torch.no_grad():
        u_final = model(x_obs, t_obs)
    plt.scatter(u_obs.numpy(), u_final.numpy(), alpha=0.5)
    plt.plot([u_obs.min(), u_obs.max()], [u_obs.min(), u_obs.max()], 'r--')
    plt.xlabel('观测值')
    plt.ylabel('预测值')
    plt.title('预测 vs 观测')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_inverse_problem.png', dpi=150)
    plt.show()
    
    return model


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有 PINN 示例"""
    print("物理信息神经网络 (PINN) 示例")
    print("=" * 60)
    
    solve_first_order_ode()
    solve_harmonic_oscillator()
    solve_heat_equation()
    solve_wave_equation()
    solve_inverse_problem()
    
    print("\n所有示例完成！")


def demo():
    """快速演示"""
    print("PINN 快速演示 - 一阶 ODE")
    print("=" * 60)
    
    model = PINN(input_dim=1, output_dim=1, hidden_dims=[32, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.linspace(0, 3, 50, requires_grad=True).reshape(-1, 1)
    x_bc = torch.zeros(1, 1)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        y = model(x)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        
        loss = (dy + y).pow(2).mean() + 10 * (model(x_bc) - 1).pow(2)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    with torch.no_grad():
        y_pred = model(x)
        y_exact = torch.exp(-x)
    
    print(f"\n最大误差: {(y_pred - y_exact).abs().max().item():.6f}")
    print("快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        main()

