#!/usr/bin/env python3
"""
线性回归完整示例

本脚本演示如何使用 PyTorch 实现线性回归和多项式回归，
包括手动实现和使用 nn.Module 的两种方式。

适合有 Python 基础的物理科研人员学习。

运行方式：
    python linear_regression.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 简单线性回归（手动实现）
# ============================================================

def simple_linear_regression_manual():
    """手动实现简单线性回归"""
    print_section("1. 简单线性回归（手动实现）")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 生成模拟数据：y = 2x + 1 + 噪声
    true_w, true_b = 2.0, 1.0
    X = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y_true = true_w * X + true_b + torch.randn_like(X) * 0.5
    
    print(f"数据形状: X={X.shape}, y={y_true.shape}")
    print(f"真实参数: w={true_w}, b={true_b}")
    
    # 初始化参数
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # 超参数
    lr = 0.01
    epochs = 100
    
    # 训练记录
    losses = []
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        y_pred = w * X + b
        
        # 计算 MSE 损失
        loss = ((y_pred - y_true) ** 2).mean()
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")
    
    print(f"\n最终结果:")
    print(f"  学习到: w={w.item():.4f}, b={b.item():.4f}")
    print(f"  真实值: w={true_w:.4f}, b={true_b:.4f}")
    print(f"  相对误差: w: {abs(w.item()-true_w)/true_w*100:.2f}%, b: {abs(b.item()-true_b)/true_b*100:.2f}%")
    
    return X, y_true, w, b, losses


# ============================================================
# 2. 使用 nn.Module 的线性回归
# ============================================================

class LinearRegressionModel(nn.Module):
    """线性回归模型"""
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def linear_regression_with_module():
    """使用 nn.Module 实现线性回归"""
    print_section("2. 使用 nn.Module 的线性回归")
    
    torch.manual_seed(42)
    
    # 多元线性回归数据
    # y = 2*x1 + 3*x2 - 1*x3 + 0.5 + 噪声
    N, d = 200, 3
    X = torch.randn(N, d)
    w_true = torch.tensor([[2.0], [3.0], [-1.0]])
    b_true = 0.5
    y_true = X @ w_true + b_true + torch.randn(N, 1) * 0.3
    
    print(f"数据形状: X={X.shape}, y={y_true.shape}")
    print(f"真实权重: {w_true.flatten().tolist()}")
    print(f"真实偏置: {b_true}")
    
    # 创建模型
    model = LinearRegressionModel(input_dim=d)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 训练
    epochs = 100
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.6f}")
    
    # 结果
    learned_w = model.linear.weight.data.flatten().tolist()
    learned_b = model.linear.bias.item()
    
    print(f"\n结果对比:")
    print(f"  真实权重: {w_true.flatten().tolist()}")
    print(f"  学习权重: [{learned_w[0]:.4f}, {learned_w[1]:.4f}, {learned_w[2]:.4f}]")
    print(f"  真实偏置: {b_true}")
    print(f"  学习偏置: {learned_b:.4f}")
    
    return model


# ============================================================
# 3. 多项式回归
# ============================================================

def polynomial_regression():
    """多项式回归：拟合非线性函数"""
    print_section("3. 多项式回归")
    
    torch.manual_seed(42)
    
    # 生成数据：y = 0.5 + 2x - 0.3x^2 + 0.1x^3 + 噪声
    x = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = 0.5 + 2*x - 0.3*x**2 + 0.1*x**3 + torch.randn_like(x) * 0.5
    
    # 特征工程：创建多项式特征 [x, x^2, x^3]
    def create_poly_features(x, degree=3):
        return torch.cat([x ** i for i in range(1, degree + 1)], dim=1)
    
    X_poly = create_poly_features(x, degree=3)
    print(f"原始特征形状: {x.shape}")
    print(f"多项式特征形状: {X_poly.shape}")
    
    # 模型（实际上还是线性回归，只是特征是非线性的）
    model = nn.Linear(3, 1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # 训练
    epochs = 500
    for epoch in range(epochs):
        y_pred = model(X_poly)
        loss = criterion(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.6f}")
    
    # 学习到的系数
    coeffs = model.weight.data.flatten().tolist()
    bias = model.bias.item()
    
    print(f"\n学习到的多项式:")
    print(f"  y = {bias:.4f} + {coeffs[0]:.4f}x + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x³")
    print(f"真实多项式:")
    print(f"  y = 0.5000 + 2.0000x + (-0.3000)x² + 0.1000x³")
    
    return x, y_true, model, X_poly


# ============================================================
# 4. 带正则化的线性回归（Ridge 回归）
# ============================================================

def ridge_regression():
    """带 L2 正则化的线性回归"""
    print_section("4. Ridge 回归（L2 正则化）")
    
    torch.manual_seed(42)
    
    # 生成数据（特征数量多于样本数量，容易过拟合）
    N, d = 50, 100  # 50 个样本，100 个特征
    X = torch.randn(N, d)
    w_true = torch.zeros(d, 1)
    w_true[:5] = torch.tensor([[1.], [2.], [-1.], [0.5], [-0.5]])  # 只有前5个特征有用
    y_true = X @ w_true + torch.randn(N, 1) * 0.1
    
    print(f"数据形状: X={X.shape}, y={y_true.shape}")
    print(f"真实有效特征: 前5个")
    
    # 普通线性回归
    model_no_reg = nn.Linear(d, 1)
    optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.01)
    
    # 带正则化的线性回归
    model_ridge = nn.Linear(d, 1)
    # weight_decay 参数就是 L2 正则化系数
    optimizer_ridge = optim.Adam(model_ridge.parameters(), lr=0.01, weight_decay=0.1)
    
    criterion = nn.MSELoss()
    
    # 训练
    epochs = 1000
    for epoch in range(epochs):
        # 无正则化
        y_pred_no_reg = model_no_reg(X)
        loss_no_reg = criterion(y_pred_no_reg, y_true)
        optimizer_no_reg.zero_grad()
        loss_no_reg.backward()
        optimizer_no_reg.step()
        
        # 有正则化
        y_pred_ridge = model_ridge(X)
        loss_ridge = criterion(y_pred_ridge, y_true)
        optimizer_ridge.zero_grad()
        loss_ridge.backward()
        optimizer_ridge.step()
    
    # 比较权重
    w_no_reg = model_no_reg.weight.data.flatten()
    w_ridge = model_ridge.weight.data.flatten()
    
    print(f"\n权重统计:")
    print(f"  无正则化: 均值={w_no_reg.mean().item():.4f}, 标准差={w_no_reg.std().item():.4f}")
    print(f"  有正则化: 均值={w_ridge.mean().item():.4f}, 标准差={w_ridge.std().item():.4f}")
    print(f"\n前5个权重（应该接近 [1, 2, -1, 0.5, -0.5]）:")
    print(f"  无正则化: {w_no_reg[:5].tolist()}")
    print(f"  有正则化: {w_ridge[:5].tolist()}")
    print(f"\n权重的 L2 范数:")
    print(f"  无正则化: {w_no_reg.norm().item():.4f}")
    print(f"  有正则化: {w_ridge.norm().item():.4f}")
    
    return model_no_reg, model_ridge


# ============================================================
# 5. 物理应用：胡克定律拟合
# ============================================================

def hookes_law_fitting():
    """物理应用：从实验数据拟合胡克定律"""
    print_section("5. 物理应用：胡克定律拟合")
    
    torch.manual_seed(42)
    
    # 模拟实验数据
    # 胡克定律：F = kx，k 是弹簧常数
    k_true = 25.0  # N/m
    
    # "实验"数据：位移和对应的力
    x = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])  # 米
    F = k_true * x + torch.randn(10) * 0.5  # 力 + 实验误差
    
    x = x.reshape(-1, 1)
    F = F.reshape(-1, 1)
    
    print("实验数据（位移 vs 力）:")
    for i in range(len(x)):
        print(f"  x = {x[i].item():.2f} m, F = {F[i].item():.2f} N")
    
    # 使用线性回归拟合（不带偏置，因为 F(0) = 0）
    model = nn.Linear(1, 1, bias=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    
    # 训练
    for epoch in range(500):
        F_pred = model(x)
        loss = criterion(F_pred, F)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    k_fitted = model.weight.item()
    
    print(f"\n拟合结果:")
    print(f"  测量弹簧常数: k = {k_fitted:.2f} N/m")
    print(f"  真实弹簧常数: k = {k_true:.2f} N/m")
    print(f"  相对误差: {abs(k_fitted - k_true) / k_true * 100:.2f}%")
    
    # 计算拟合优度 R²
    ss_res = ((F - model(x)) ** 2).sum()
    ss_tot = ((F - F.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    print(f"  拟合优度 R²: {r_squared.item():.6f}")
    
    return x, F, model


# ============================================================
# 6. 可视化
# ============================================================

def visualize_results(x1, y1, w, b, losses, x_poly, y_poly, model_poly, X_poly_features):
    """可视化所有结果"""
    print_section("6. 生成可视化图表")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1：简单线性回归
    ax1 = axes[0, 0]
    ax1.scatter(x1.numpy(), y1.numpy(), alpha=0.5, label='数据点')
    x_line = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y_line = w.item() * x_line + b.item()
    ax1.plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, 
             label=f'拟合线: y = {w.item():.2f}x + {b.item():.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('简单线性回归')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：损失曲线
    ax2 = axes[0, 1]
    ax2.plot(losses, 'b-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('训练损失曲线')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 图3：多项式回归
    ax3 = axes[1, 0]
    ax3.scatter(x_poly.numpy(), y_poly.numpy(), alpha=0.5, label='数据点')
    x_smooth = torch.linspace(-3, 3, 200).reshape(-1, 1)
    X_smooth_poly = torch.cat([x_smooth ** i for i in range(1, 4)], dim=1)
    y_smooth = model_poly(X_smooth_poly).detach()
    ax3.plot(x_smooth.numpy(), y_smooth.numpy(), 'r-', linewidth=2, label='拟合曲线')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('多项式回归')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4：物理视角 - 损失函数等高线
    ax4 = axes[1, 1]
    w_range = np.linspace(0, 4, 100)
    b_range = np.linspace(-1, 3, 100)
    W, B = np.meshgrid(w_range, b_range)
    
    # 计算损失面（使用简单线性回归的数据）
    X_np = x1.numpy()
    y_np = y1.numpy()
    Loss = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            y_pred = W[j, i] * X_np + B[j, i]
            Loss[j, i] = ((y_pred - y_np) ** 2).mean()
    
    contour = ax4.contour(W, B, Loss, levels=20, cmap='viridis')
    ax4.clabel(contour, inline=True, fontsize=8)
    ax4.plot(w.item(), b.item(), 'r*', markersize=15, label='最优解')
    ax4.plot(2.0, 1.0, 'g^', markersize=10, label='真实值')
    ax4.set_xlabel('w')
    ax4.set_ylabel('b')
    ax4.set_title('损失函数等高线（物理视角：势能面）')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=150)
    print("图表已保存为 linear_regression_results.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" PyTorch 线性回归完整示例")
    print("="*60)
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 运行所有示例
    x1, y1, w, b, losses = simple_linear_regression_manual()
    linear_regression_with_module()
    x_poly, y_poly, model_poly, X_poly = polynomial_regression()
    ridge_regression()
    hookes_law_fitting()
    
    # 可视化
    visualize_results(x1, y1, w, b, losses, x_poly, y_poly, model_poly, X_poly)
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 线性回归是最简单的神经网络（单层，无激活函数）")
    print("  2. nn.Linear 实现 y = Wx + b")
    print("  3. 多项式回归 = 特征工程 + 线性回归")
    print("  4. L2 正则化通过 weight_decay 参数实现")
    print("  5. 损失函数可以看作物理中的势能面")
    print()


if __name__ == "__main__":
    main()

