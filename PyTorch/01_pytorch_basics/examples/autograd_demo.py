#!/usr/bin/env python3
"""
PyTorch 自动微分 (Autograd) 示例代码

本脚本演示 PyTorch 的自动微分系统，这是神经网络训练的核心。
从物理视角理解：计算梯度类似于计算势能面上的力（负梯度方向）。

运行方式：
    python autograd_demo.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def basic_gradient():
    """基本梯度计算示例"""
    print_section("1. 基本梯度计算")
    
    # 创建需要梯度的张量
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    # 前向传播：构建计算图
    z = x * y           # z = 6
    w = z + x           # w = 8
    v = w ** 2          # v = 64
    
    print(f"计算过程: v = (x*y + x)^2")
    print(f"  x = {x.item()}")
    print(f"  y = {y.item()}")
    print(f"  z = x*y = {z.item()}")
    print(f"  w = z+x = {w.item()}")
    print(f"  v = w^2 = {v.item()}")
    
    # 反向传播：计算梯度
    v.backward()
    
    # 解析验证：
    # v = (xy + x)^2
    # ∂v/∂x = 2(xy + x)(y + 1) = 2 * 8 * 4 = 64
    # ∂v/∂y = 2(xy + x) * x = 2 * 8 * 2 = 32
    
    print(f"\n梯度计算结果:")
    print(f"  ∂v/∂x = {x.grad.item()}")
    print(f"  ∂v/∂y = {y.grad.item()}")
    print(f"\n验证 (解析解):")
    print(f"  ∂v/∂x = 2(xy+x)(y+1) = 2*8*4 = 64 ✓")
    print(f"  ∂v/∂y = 2(xy+x)*x = 2*8*2 = 32 ✓")


def gradient_accumulation():
    """梯度累积演示"""
    print_section("2. 梯度累积")
    
    x = torch.tensor([1.0], requires_grad=True)
    
    print("梯度会累积！每次 backward() 会将梯度加到现有值上：")
    
    # 第一次
    y1 = x * 2
    y1.backward()
    print(f"  第1次 backward(): x.grad = {x.grad.item()}")
    
    # 第二次（梯度累积）
    y2 = x * 3
    y2.backward()
    print(f"  第2次 backward(): x.grad = {x.grad.item()} (2 + 3 = 5)")
    
    # 清零梯度
    x.grad.zero_()
    print(f"  清零后: x.grad = {x.grad.item()}")
    
    # 第三次
    y3 = x * 4
    y3.backward()
    print(f"  第3次 backward(): x.grad = {x.grad.item()}")
    
    print("\n重要：在训练循环中，每次迭代前都要清零梯度！")


def no_grad_demo():
    """torch.no_grad() 演示"""
    print_section("3. 停止梯度追踪")
    
    x = torch.tensor([1.0], requires_grad=True)
    
    # 正常情况
    y = x * 2
    print(f"正常计算: y.requires_grad = {y.requires_grad}")
    
    # 使用 no_grad()
    with torch.no_grad():
        y_no_grad = x * 2
        print(f"no_grad 内: y.requires_grad = {y_no_grad.requires_grad}")
    
    # 使用 detach()
    y_detached = y.detach()
    print(f"detach() 后: y_detached.requires_grad = {y_detached.requires_grad}")
    
    print("\n用途：")
    print("  - 模型推理时使用 no_grad() 节省内存和加速")
    print("  - 需要从计算图中分离张量时使用 detach()")


def higher_order_derivatives():
    """高阶导数计算"""
    print_section("4. 高阶导数")
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # y = x^4
    y = x ** 4
    print(f"函数: y = x^4, x = {x.item()}")
    
    # 一阶导数: dy/dx = 4x^3
    dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"一阶导数: dy/dx = 4x³ = {dy_dx.item()} (理论值: {4 * 2**3})")
    
    # 二阶导数: d²y/dx² = 12x²
    d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
    print(f"二阶导数: d²y/dx² = 12x² = {d2y_dx2.item()} (理论值: {12 * 2**2})")
    
    # 三阶导数: d³y/dx³ = 24x
    d3y_dx3 = torch.autograd.grad(d2y_dx2, x, create_graph=True)[0]
    print(f"三阶导数: d³y/dx³ = 24x = {d3y_dx3.item()} (理论值: {24 * 2})")
    
    # 四阶导数: d⁴y/dx⁴ = 24
    d4y_dx4 = torch.autograd.grad(d3y_dx3, x)[0]
    print(f"四阶导数: d⁴y/dx⁴ = 24 = {d4y_dx4.item()}")


def multivariate_gradient():
    """多变量函数梯度"""
    print_section("5. 多变量函数梯度")
    
    # f(x, y, z) = x²y + xyz + z³
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    z = torch.tensor([3.0], requires_grad=True)
    
    f = x**2 * y + x * y * z + z**3
    
    print(f"函数: f(x, y, z) = x²y + xyz + z³")
    print(f"点: ({x.item()}, {y.item()}, {z.item()})")
    print(f"函数值: f = {f.item()}")
    
    f.backward()
    
    # 解析验证：
    # ∂f/∂x = 2xy + yz = 2*1*2 + 2*3 = 10
    # ∂f/∂y = x² + xz = 1 + 3 = 4
    # ∂f/∂z = xy + 3z² = 2 + 27 = 29
    
    print(f"\n梯度 (∇f):")
    print(f"  ∂f/∂x = {x.grad.item()} (理论: 2xy + yz = 10)")
    print(f"  ∂f/∂y = {y.grad.item()} (理论: x² + xz = 4)")
    print(f"  ∂f/∂z = {z.grad.item()} (理论: xy + 3z² = 29)")


def gradient_descent_optimization():
    """梯度下降优化示例"""
    print_section("6. 梯度下降优化")
    
    def potential(x: torch.Tensor) -> torch.Tensor:
        """双势阱势能函数: V(x) = x⁴ - 10x² + 5"""
        return x**4 - 10*x**2 + 5
    
    # 初始位置
    x = torch.tensor([3.0], requires_grad=True)
    
    # 优化参数
    learning_rate = 0.01
    n_steps = 300
    
    # 记录轨迹
    trajectory = [x.item()]
    energies = [potential(x).item()]
    
    print("双势阱势能: V(x) = x⁴ - 10x² + 5")
    print(f"初始位置: x = {x.item():.4f}")
    print(f"初始势能: V = {energies[0]:.4f}")
    print(f"学习率: {learning_rate}")
    print(f"迭代次数: {n_steps}")
    
    # 梯度下降循环
    for step in range(n_steps):
        # 前向传播
        V = potential(x)
        
        # 反向传播
        V.backward()
        
        # 更新位置（不追踪梯度）
        with torch.no_grad():
            x -= learning_rate * x.grad
        
        # 清零梯度
        x.grad.zero_()
        
        # 记录
        trajectory.append(x.item())
        energies.append(potential(x).item())
    
    print(f"\n最终位置: x = {trajectory[-1]:.4f}")
    print(f"最终势能: V = {energies[-1]:.4f}")
    
    # 理论极值点
    # dV/dx = 4x³ - 20x = 4x(x² - 5) = 0
    # x = 0, ±√5 ≈ ±2.236
    print(f"\n理论极小值点: x = ±√5 ≈ ±{5**0.5:.4f}")
    print(f"理论最小势能: V(±√5) = {potential(torch.tensor([5**0.5])).item():.4f}")
    
    return trajectory, energies


def harmonic_oscillator_demo():
    """物理应用：简谐振子"""
    print_section("7. 物理应用：简谐振子势能分析")
    
    # 简谐振子势能: V(x) = 1/2 * k * x²
    k = 2.0  # 弹簧常数
    
    x = torch.tensor([3.0], requires_grad=True)
    
    V = 0.5 * k * x**2
    
    print(f"简谐振子势能: V(x) = ½kx²")
    print(f"弹簧常数: k = {k}")
    print(f"位置: x = {x.item()}")
    print(f"势能: V = {V.item()}")
    
    # 力 F = -dV/dx
    grad_V = torch.autograd.grad(V, x, create_graph=True)[0]
    F = -grad_V
    print(f"\n力: F = -dV/dx = -kx = {F.item()}")
    print(f"验证: -kx = -{k}*{x.item()} = {-k*x.item()}")
    
    # 力常数 = d²V/dx²
    d2V_dx2 = torch.autograd.grad(grad_V, x)[0]
    print(f"\n力常数: d²V/dx² = k = {d2V_dx2.item()}")


def lennard_jones_demo():
    """物理应用：Lennard-Jones 势"""
    print_section("8. 物理应用：Lennard-Jones 势")
    
    # LJ 势: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    epsilon = 1.0  # 势阱深度
    sigma = 1.0    # 特征长度
    
    r = torch.tensor([1.5], requires_grad=True)
    
    def lj_potential(r):
        r6 = (sigma / r) ** 6
        r12 = r6 ** 2
        return 4 * epsilon * (r12 - r6)
    
    V = lj_potential(r)
    
    print(f"Lennard-Jones 势: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]")
    print(f"参数: ε = {epsilon}, σ = {sigma}")
    print(f"距离: r = {r.item()}")
    print(f"势能: V = {V.item():.4f}")
    
    # 力 F = -dV/dr
    V.backward()
    F = -r.grad
    print(f"\n力: F = -dV/dr = {F.item():.4f}")
    
    # 理论平衡位置: dV/dr = 0 => r = 2^(1/6) * σ
    r_eq = 2**(1/6) * sigma
    print(f"\n平衡位置: r_eq = 2^(1/6)σ = {r_eq:.4f}")
    print(f"平衡时势能: V(r_eq) = -ε = {lj_potential(torch.tensor([r_eq])).item():.4f}")


def jacobian_demo():
    """雅可比矩阵计算"""
    print_section("9. 雅可比矩阵")
    
    from torch.autograd.functional import jacobian
    
    # 向量函数 f: R² -> R²
    def f(x):
        """
        f([x1, x2]) = [x1*x2, x1² + x2²]
        """
        return torch.stack([
            x[0] * x[1],
            x[0]**2 + x[1]**2
        ])
    
    x = torch.tensor([2.0, 3.0])
    
    print(f"向量函数: f([x₁, x₂]) = [x₁x₂, x₁² + x₂²]")
    print(f"点: x = {x.tolist()}")
    print(f"函数值: f(x) = {f(x).tolist()}")
    
    # 计算雅可比矩阵
    J = jacobian(f, x)
    
    print(f"\n雅可比矩阵 J = ∂f/∂x:")
    print(f"  [[∂f₁/∂x₁, ∂f₁/∂x₂],    [[x₂, x₁],")
    print(f"   [∂f₂/∂x₁, ∂f₂/∂x₂]]  =  [2x₁, 2x₂]]")
    print(f"\n数值结果:")
    print(f"  {J}")
    print(f"\n验证:")
    print(f"  J[0,0] = x₂ = {x[1].item()}")
    print(f"  J[0,1] = x₁ = {x[0].item()}")
    print(f"  J[1,0] = 2x₁ = {2*x[0].item()}")
    print(f"  J[1,1] = 2x₂ = {2*x[1].item()}")


def custom_autograd_function():
    """自定义自动微分函数"""
    print_section("10. 自定义自动微分函数")
    
    class MySoftplus(torch.autograd.Function):
        """
        自定义 Softplus 函数: f(x) = log(1 + exp(x))
        导数: f'(x) = exp(x) / (1 + exp(x)) = sigmoid(x)
        """
        
        @staticmethod
        def forward(ctx, x):
            # 保存反向传播需要的值
            exp_x = torch.exp(x)
            ctx.save_for_backward(exp_x)
            return torch.log(1 + exp_x)
        
        @staticmethod
        def backward(ctx, grad_output):
            exp_x, = ctx.saved_tensors
            # 导数: exp(x) / (1 + exp(x))
            grad_input = grad_output * exp_x / (1 + exp_x)
            return grad_input
    
    # 测试自定义函数
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # 使用自定义函数
    y = MySoftplus.apply(x)
    loss = y.sum()
    loss.backward()
    
    print("自定义 Softplus 函数: f(x) = log(1 + exp(x))")
    print(f"输入: {x.data.tolist()}")
    print(f"输出: {y.data.tolist()}")
    print(f"梯度: {x.grad.tolist()}")
    
    # 与 PyTorch 内置函数对比
    x2 = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y2 = torch.nn.functional.softplus(x2)
    y2.sum().backward()
    
    print(f"\nPyTorch 内置 softplus 梯度: {x2.grad.tolist()}")
    print(f"梯度一致: {torch.allclose(x.grad, x2.grad)}")


def physics_informed_gradient():
    """物理应用：满足物理约束的梯度计算"""
    print_section("11. 物理应用：热传导方程残差")
    
    # 一维热传导方程: ∂u/∂t = α * ∂²u/∂x²
    # 我们用神经网络的输出来近似 u(x, t)
    # 这里用简单的参数化函数演示
    
    alpha = 0.1  # 热扩散系数
    
    # 假设 u(x, t) = exp(-t) * sin(x)（这是某些边界条件下的解析解）
    def u_approx(x, t):
        return torch.exp(-t) * torch.sin(x)
    
    # 采样点
    x = torch.tensor([1.0], requires_grad=True)
    t = torch.tensor([0.5], requires_grad=True)
    
    # 计算 u
    u = u_approx(x, t)
    
    print("一维热传导方程: ∂u/∂t = α ∂²u/∂x²")
    print(f"热扩散系数: α = {alpha}")
    print(f"采样点: (x, t) = ({x.item()}, {t.item()})")
    print(f"u(x, t) = {u.item():.6f}")
    
    # 计算 ∂u/∂t
    du_dt = torch.autograd.grad(u, t, create_graph=True)[0]
    print(f"\n∂u/∂t = {du_dt.item():.6f}")
    
    # 计算 ∂u/∂x
    du_dx = torch.autograd.grad(u, x, create_graph=True)[0]
    print(f"∂u/∂x = {du_dx.item():.6f}")
    
    # 计算 ∂²u/∂x²
    d2u_dx2 = torch.autograd.grad(du_dx, x, create_graph=True)[0]
    print(f"∂²u/∂x² = {d2u_dx2.item():.6f}")
    
    # PDE 残差: ∂u/∂t - α * ∂²u/∂x²
    residual = du_dt - alpha * d2u_dx2
    print(f"\nPDE 残差: ∂u/∂t - α∂²u/∂x² = {residual.item():.6f}")
    
    # 对于真解，残差应该为 0（这里因为 α ≠ 1，所以不是零）
    # 真解对应 α = 1 的情况
    print(f"（注：当 α = 1 时残差为 0，这里 α = {alpha}）")


def plot_gradient_descent(trajectory: List[float], energies: List[float]) -> None:
    """可视化梯度下降过程（可选）"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 势能面
        x_plot = np.linspace(-4, 4, 200)
        V_plot = x_plot**4 - 10*x_plot**2 + 5
        
        axes[0].plot(x_plot, V_plot, 'b-', linewidth=2, label='V(x) = x⁴ - 10x² + 5')
        axes[0].plot(trajectory, energies, 'r.-', markersize=2, alpha=0.7, label='优化轨迹')
        axes[0].scatter(trajectory[0], energies[0], c='green', s=100, zorder=5, label='起点')
        axes[0].scatter(trajectory[-1], energies[-1], c='red', s=100, zorder=5, label='终点')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('V(x)')
        axes[0].set_title('势能面与优化轨迹')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 能量随迭代变化
        axes[1].plot(energies, 'b-', linewidth=1)
        axes[1].set_xlabel('迭代次数')
        axes[1].set_ylabel('势能 V(x)')
        axes[1].set_title('势能随迭代的变化')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_descent.png', dpi=150)
        print("\n图形已保存为 gradient_descent.png")
        
    except ImportError:
        print("\n（matplotlib 未安装，跳过可视化）")


def main():
    """主函数"""
    print("\n" + "="*60)
    print(" PyTorch 自动微分 (Autograd) 示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 运行所有示例
    basic_gradient()
    gradient_accumulation()
    no_grad_demo()
    higher_order_derivatives()
    multivariate_gradient()
    trajectory, energies = gradient_descent_optimization()
    harmonic_oscillator_demo()
    lennard_jones_demo()
    jacobian_demo()
    custom_autograd_function()
    physics_informed_gradient()
    
    # 可视化（可选）
    plot_gradient_descent(trajectory, energies)
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点：")
    print("  1. requires_grad=True 启用梯度追踪")
    print("  2. .backward() 计算梯度")
    print("  3. 每次迭代前要 .grad.zero_() 清零梯度")
    print("  4. 推理时使用 torch.no_grad() 节省内存")
    print("  5. create_graph=True 可计算高阶导数")
    print()


if __name__ == "__main__":
    main()

