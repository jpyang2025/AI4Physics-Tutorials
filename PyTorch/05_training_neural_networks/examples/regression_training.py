#!/usr/bin/env python3
"""
回归问题训练示例

本示例展示回归问题的完整训练流程，包括：
1. 函数拟合
2. 物理系统参数估计
3. 微分方程求解（PINN 风格）

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. 数据集定义
# =============================================================================

class FunctionDataset(Dataset):
    """函数拟合数据集"""
    
    def __init__(self, func, x_range, n_samples, noise_level=0.1):
        """
        Args:
            func: 目标函数 y = f(x)
            x_range: x 范围 (min, max)
            n_samples: 样本数量
            noise_level: 噪声水平
        """
        self.x = torch.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
        self.y_true = func(self.x)
        self.y = self.y_true + noise_level * torch.randn_like(self.y_true)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HarmonicOscillatorDataset(Dataset):
    """谐振子数据集 - 根据轨迹推断参数"""
    
    def __init__(self, n_samples=1000, t_points=100, noise_level=0.05):
        """
        生成形如 x(t) = A*cos(ω*t + φ) 的轨迹
        目标是从轨迹推断 (A, ω, φ)
        """
        self.t = torch.linspace(0, 10, t_points)
        self.trajectories = []
        self.parameters = []
        
        for _ in range(n_samples):
            A = np.random.uniform(0.5, 2.0)
            omega = np.random.uniform(0.5, 3.0)
            phi = np.random.uniform(0, 2 * np.pi)
            
            x = A * torch.cos(omega * self.t + phi)
            x = x + noise_level * torch.randn_like(x)
            
            self.trajectories.append(x)
            self.parameters.append(torch.tensor([A, omega, phi]))
        
        self.trajectories = torch.stack(self.trajectories)
        self.parameters = torch.stack(self.parameters)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.parameters[idx]


# =============================================================================
# 2. 模型定义
# =============================================================================

class RegressionMLP(nn.Module):
    """回归 MLP"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()  # Tanh 对回归问题效果好
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class PhysicsNet(nn.Module):
    """物理约束网络 - 用于求解 ODE"""
    
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.net(t)


# =============================================================================
# 3. 训练函数
# =============================================================================

def train_regression_model(model, train_loader, val_loader, 
                          criterion, optimizer, num_epochs, device):
    """训练回归模型"""
    
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * x.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
    
    return history


# =============================================================================
# 4. 示例：函数拟合
# =============================================================================

def example_function_fitting():
    """示例：拟合复杂函数"""
    print("\n" + "=" * 60)
    print("示例 1: 函数拟合")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 目标函数
    def target_function(x):
        return torch.sin(2 * x) * torch.exp(-0.1 * x**2) + 0.5 * torch.cos(5 * x)
    
    # 创建数据集
    dataset = FunctionDataset(target_function, (-5, 5), 1000, noise_level=0.05)
    
    # 划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 模型
    model = RegressionMLP(1, 1, hidden_dims=[64, 64, 64])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    history = train_regression_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs=100, device=device
    )
    
    # 可视化
    model.eval()
    x_test = torch.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = target_function(x_test)
    
    with torch.no_grad():
        y_pred = model(x_test.to(device)).cpu()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(dataset.x.numpy(), dataset.y.numpy(), alpha=0.3, label='数据点')
    plt.plot(x_test.numpy(), y_true.numpy(), 'g-', linewidth=2, label='真实函数')
    plt.plot(x_test.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='拟合结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('函数拟合结果')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('训练历史')
    
    plt.tight_layout()
    plt.savefig('function_fitting.png')
    plt.show()
    
    print(f"\n最终验证损失: {history['val_loss'][-1]:.6f}")


# =============================================================================
# 5. 示例：物理参数估计
# =============================================================================

def example_parameter_estimation():
    """示例：从谐振子轨迹估计参数"""
    print("\n" + "=" * 60)
    print("示例 2: 谐振子参数估计")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集
    dataset = HarmonicOscillatorDataset(n_samples=2000)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 模型：从 100 维轨迹预测 3 个参数
    model = RegressionMLP(100, 3, hidden_dims=[128, 64, 32])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    history = train_regression_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs=100, device=device
    )
    
    # 评估
    model.eval()
    model = model.to(device)
    
    test_trajectories, test_params = dataset[1900:1910]
    test_trajectories = test_trajectories.to(device)
    
    with torch.no_grad():
        pred_params = model(test_trajectories)
    
    print("\n参数估计结果（前5个样本）:")
    print("-" * 60)
    print(f"{'真实 (A, ω, φ)':<30} {'预测 (A, ω, φ)':<30}")
    print("-" * 60)
    
    for i in range(5):
        true = test_params[i].numpy()
        pred = pred_params[i].cpu().numpy()
        print(f"({true[0]:.3f}, {true[1]:.3f}, {true[2]:.3f})" + " " * 10 +
              f"({pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f})")
    
    # 可视化轨迹预测
    plt.figure(figsize=(12, 4))
    
    t = dataset.t.numpy()
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        
        trajectory = test_trajectories[i].cpu().numpy()
        true_A, true_omega, true_phi = test_params[i].numpy()
        pred_A, pred_omega, pred_phi = pred_params[i].cpu().numpy()
        
        # 重构轨迹
        reconstructed = pred_A * np.cos(pred_omega * t + pred_phi)
        
        plt.plot(t, trajectory, 'b-', alpha=0.5, label='实测轨迹')
        plt.plot(t, reconstructed, 'r--', label='重构轨迹')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend(fontsize=8)
        plt.title(f'样本 {i+1}')
    
    plt.tight_layout()
    plt.savefig('parameter_estimation.png')
    plt.show()


# =============================================================================
# 6. 示例：PINN 求解 ODE
# =============================================================================

def example_pinn_ode():
    """
    示例：用 PINN 求解常微分方程
    
    问题：y'' + y = 0
    边界条件：y(0) = 0, y(π) = 0
    解析解：y(x) = A*sin(x)
    """
    print("\n" + "=" * 60)
    print("示例 3: PINN 求解 ODE (y'' + y = 0)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 网络
    class PINNModel(nn.Module):
        """满足边界条件的 PINN"""
        
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            # 自动满足边界条件 y(0) = 0, y(π) = 0
            # y = x * (π - x) * NN(x)
            return x * (torch.pi - x) * self.net(x)
    
    model = PINNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 内部配点
    x_interior = torch.linspace(0.01, torch.pi - 0.01, 100).reshape(-1, 1).to(device)
    x_interior.requires_grad = True
    
    # 训练
    losses = []
    
    for epoch in tqdm(range(5000), desc='Training PINN'):
        optimizer.zero_grad()
        
        # 计算 y, y', y''
        y = model(x_interior)
        
        y_x = torch.autograd.grad(
            y, x_interior,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        y_xx = torch.autograd.grad(
            y_x, x_interior,
            grad_outputs=torch.ones_like(y_x),
            create_graph=True
        )[0]
        
        # PDE 残差：y'' + y = 0
        residual = y_xx + y
        loss = (residual ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}, Residual Loss: {loss.item():.6f}")
    
    # 评估
    model.eval()
    x_test = torch.linspace(0, torch.pi, 100).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        y_pred = model(x_test)
    
    # 归一化后比较（因为真实解有任意振幅）
    y_pred_normalized = y_pred / y_pred.max()
    y_exact = torch.sin(x_test)
    
    error = (y_pred_normalized - y_exact).abs().max().item()
    print(f"\n最大归一化误差: {error:.6f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_test.cpu().numpy(), y_pred.cpu().numpy(), 'r-', 
             linewidth=2, label='PINN 解')
    plt.plot(x_test.cpu().numpy(), y_exact.cpu().numpy() * y_pred.max().item(), 
             'b--', linewidth=2, label='解析解 (缩放)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('ODE 解')
    
    plt.subplot(1, 3, 2)
    plt.plot(x_test.cpu().numpy(), 
             (y_pred_normalized - y_exact).cpu().numpy(), 'k-')
    plt.xlabel('x')
    plt.ylabel('误差')
    plt.title('归一化误差')
    
    plt.subplot(1, 3, 3)
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Loss')
    plt.title('训练损失')
    
    plt.tight_layout()
    plt.savefig('pinn_ode.png')
    plt.show()


# =============================================================================
# 7. 示例：热传导方程
# =============================================================================

def example_heat_equation():
    """
    示例：用 PINN 求解热传导方程
    
    问题：∂u/∂t = α * ∂²u/∂x²
    初始条件：u(x, 0) = sin(πx)
    边界条件：u(0, t) = u(1, t) = 0
    解析解：u(x, t) = sin(πx) * exp(-α*π²*t)
    """
    print("\n" + "=" * 60)
    print("示例 4: PINN 求解热传导方程")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    alpha = 0.1  # 热扩散系数
    
    class HeatPINN(nn.Module):
        """热传导方程 PINN"""
        
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64),  # 输入 (x, t)
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x, t):
            # 输入拼接
            inputs = torch.cat([x, t], dim=1)
            u = self.net(inputs)
            
            # 自动满足边界条件
            # u = x * (1 - x) * NN(x, t)
            # 同时添加初始条件的影响
            return x * (1 - x) * u
    
    model = HeatPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 配点
    n_interior = 1000
    n_initial = 100
    
    # 内部点
    x_interior = torch.rand(n_interior, 1).to(device)
    t_interior = torch.rand(n_interior, 1).to(device) * 0.5  # t ∈ [0, 0.5]
    x_interior.requires_grad = True
    t_interior.requires_grad = True
    
    # 初始条件点
    x_initial = torch.linspace(0, 1, n_initial).reshape(-1, 1).to(device)
    t_initial = torch.zeros(n_initial, 1).to(device)
    u_initial = torch.sin(torch.pi * x_initial)
    
    # 训练
    losses = {'pde': [], 'ic': [], 'total': []}
    
    for epoch in tqdm(range(10000), desc='Training Heat PINN'):
        optimizer.zero_grad()
        
        # PDE 残差
        u = model(x_interior, t_interior)
        
        u_t = torch.autograd.grad(
            u, t_interior,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x_interior,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x_interior,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        pde_residual = u_t - alpha * u_xx
        loss_pde = (pde_residual ** 2).mean()
        
        # 初始条件残差
        u_pred_initial = model(x_initial, t_initial)
        loss_ic = ((u_pred_initial - u_initial) ** 2).mean()
        
        # 总损失
        loss = loss_pde + 10 * loss_ic
        
        loss.backward()
        optimizer.step()
        
        losses['pde'].append(loss_pde.item())
        losses['ic'].append(loss_ic.item())
        losses['total'].append(loss.item())
        
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1}: PDE = {loss_pde.item():.6f}, "
                  f"IC = {loss_ic.item():.6f}")
    
    # 评估
    model.eval()
    
    # 创建网格
    x_grid = torch.linspace(0, 1, 50)
    t_grid = torch.linspace(0, 0.5, 50)
    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
    
    x_test = X.reshape(-1, 1).to(device)
    t_test = T.reshape(-1, 1).to(device)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test).reshape(50, 50)
    
    # 解析解
    u_exact = torch.sin(torch.pi * X) * torch.exp(-alpha * torch.pi**2 * T)
    
    error = (u_pred.cpu() - u_exact).abs()
    print(f"\n最大误差: {error.max().item():.6f}")
    print(f"平均误差: {error.mean().item():.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].pcolormesh(X.numpy(), T.numpy(), u_pred.cpu().numpy(), 
                              shading='auto', cmap='hot')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('PINN 预测')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(X.numpy(), T.numpy(), u_exact.numpy(), 
                              shading='auto', cmap='hot')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('解析解')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].pcolormesh(X.numpy(), T.numpy(), error.numpy(), 
                              shading='auto', cmap='Blues')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title('绝对误差')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('heat_equation.png')
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("回归问题训练示例")
    print("=" * 60)
    
    example_function_fitting()
    example_parameter_estimation()
    example_pinn_ode()
    example_heat_equation()
    
    print("\n所有示例运行完成!")


def demo():
    """快速演示"""
    print("回归训练快速演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 简单函数拟合
    def target_func(x):
        return torch.sin(x)
    
    x = torch.linspace(0, 2*torch.pi, 100).reshape(-1, 1)
    y = target_func(x) + 0.1 * torch.randn_like(target_func(x))
    
    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    x, y = x.to(device), y.to(device)
    
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    print("\n快速演示完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        main()

