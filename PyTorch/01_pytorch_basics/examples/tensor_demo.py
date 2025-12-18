#!/usr/bin/env python3
"""
PyTorch 张量基础示例代码

本脚本演示 PyTorch 张量的创建、操作和常用功能。
适合有 Python/NumPy 基础的物理科研人员学习。

运行方式：
    python tensor_demo.py
"""

import torch
import numpy as np
import time


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def basic_tensor_creation():
    """演示张量的基本创建方法"""
    print_section("1. 张量创建")
    
    # 从 Python 列表创建
    v = torch.tensor([1.0, 2.0, 3.0])
    print(f"从列表创建向量: {v}")
    print(f"  形状: {v.shape}, 数据类型: {v.dtype}")
    
    # 创建矩阵
    M = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    print(f"\n从列表创建矩阵:\n{M}")
    print(f"  形状: {M.shape}")
    
    # 使用工厂函数
    print("\n使用工厂函数:")
    print(f"  torch.zeros(3, 4):\n{torch.zeros(3, 4)}")
    print(f"  torch.ones(2, 3):\n{torch.ones(2, 3)}")
    print(f"  torch.eye(3):\n{torch.eye(3)}")
    print(f"  torch.rand(2, 3) (均匀分布):\n{torch.rand(2, 3)}")
    print(f"  torch.randn(2, 3) (正态分布):\n{torch.randn(2, 3)}")
    
    # 等差数列 - 物理中常用于创建网格
    x = torch.linspace(0, 2*torch.pi, 10)
    print(f"\ntorch.linspace(0, 2π, 10):\n{x}")
    
    # arange
    indices = torch.arange(0, 10, 2)
    print(f"torch.arange(0, 10, 2): {indices}")


def data_types_demo():
    """演示数据类型"""
    print_section("2. 数据类型")
    
    # 不同数据类型
    x_float32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    x_float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x_int64 = torch.tensor([1, 2], dtype=torch.int64)
    
    print(f"float32: {x_float32}, dtype={x_float32.dtype}")
    print(f"float64: {x_float64}, dtype={x_float64.dtype}")
    print(f"int64: {x_int64}, dtype={x_int64.dtype}")
    
    # 复数张量 - 物理中用于波函数、傅里叶变换
    psi = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
    print(f"\n复数张量 (波函数): {psi}")
    print(f"  实部: {psi.real}")
    print(f"  虚部: {psi.imag}")
    print(f"  模: {torch.abs(psi)}")
    print(f"  相位: {torch.angle(psi)}")
    
    # 类型转换
    x = torch.tensor([1.5, 2.7, 3.9])
    print(f"\n类型转换:")
    print(f"  原始 (float32): {x}")
    print(f"  转为 int32: {x.to(torch.int32)}")
    print(f"  转为 float64: {x.double()}")


def device_demo():
    """演示设备（CPU/GPU）"""
    print_section("3. 设备 (CPU/GPU)")
    
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 在不同设备上创建张量
    x_cpu = torch.randn(1000, 1000)
    print(f"\nCPU 张量设备: {x_cpu.device}")
    
    x_device = x_cpu.to(device)
    print(f"移动到 {device} 后: {x_device.device}")
    
    # GPU 加速演示（如果可用）
    if torch.cuda.is_available():
        print("\n--- GPU 加速演示 ---")
        size = 3000
        
        # CPU 计算
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"CPU 矩阵乘法时间: {cpu_time:.4f} 秒")
        
        # GPU 计算
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # 预热
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU 矩阵乘法时间: {gpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.1f}x")


def numpy_interop_demo():
    """演示与 NumPy 的互操作"""
    print_section("4. NumPy 互操作")
    
    # NumPy -> PyTorch
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(f"NumPy 数组:\n{np_array}")
    
    # 方法1: torch.tensor (创建副本)
    t1 = torch.tensor(np_array)
    print(f"\ntorch.tensor() - 创建副本:")
    print(f"  张量: {t1}")
    
    # 方法2: torch.from_numpy (共享内存)
    t2 = torch.from_numpy(np_array.copy())
    print(f"\ntorch.from_numpy() - 共享内存:")
    print(f"  张量: {t2}")
    
    # PyTorch -> NumPy
    tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    np_from_tensor = tensor.numpy()
    print(f"\n张量转 NumPy:\n{np_from_tensor}")


def tensor_operations_demo():
    """演示张量运算"""
    print_section("5. 张量运算")
    
    # 基本运算
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 2.0, 2.0, 2.0])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")
    print(f"x ** y = {x ** y}")
    
    # 数学函数
    print(f"\n数学函数:")
    print(f"  sqrt(x) = {torch.sqrt(x)}")
    print(f"  exp(x) = {torch.exp(x)}")
    print(f"  sin(x) = {torch.sin(x)}")
    
    # 聚合运算
    M = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(f"\n矩阵 M:\n{M}")
    print(f"  总和: {M.sum()}")
    print(f"  均值: {M.mean()}")
    print(f"  按行求和: {M.sum(dim=1)}")
    print(f"  按列求和: {M.sum(dim=0)}")
    
    # 矩阵运算
    A = torch.randn(3, 4)
    B = torch.randn(4, 2)
    C = A @ B  # 矩阵乘法
    print(f"\n矩阵乘法: ({A.shape}) @ ({B.shape}) = {C.shape}")


def linear_algebra_demo():
    """演示线性代数运算"""
    print_section("6. 线性代数")
    
    # 创建可逆矩阵
    A = torch.randn(4, 4)
    A = A @ A.T + torch.eye(4)  # 确保正定
    
    print(f"矩阵 A (4x4 正定矩阵)")
    
    # 行列式
    det = torch.linalg.det(A)
    print(f"  行列式: {det:.4f}")
    
    # 迹
    trace = torch.trace(A)
    print(f"  迹: {trace:.4f}")
    
    # 逆矩阵
    A_inv = torch.linalg.inv(A)
    identity_check = A @ A_inv
    print(f"  A @ A^(-1) ≈ I: {torch.allclose(identity_check, torch.eye(4), atol=1e-5)}")
    
    # 特征值
    eigenvalues = torch.linalg.eigvalsh(A)
    print(f"  特征值: {eigenvalues}")
    
    # 解线性方程组
    b = torch.randn(4)
    x = torch.linalg.solve(A, b)
    print(f"  解 Ax = b: 残差 = {torch.norm(A @ x - b):.2e}")


def broadcasting_demo():
    """演示广播机制"""
    print_section("7. 广播机制")
    
    # 标量广播
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"矩阵:\n{x}")
    print(f"矩阵 + 10:\n{x + 10}")
    
    # 向量广播
    row_vec = torch.tensor([100, 200, 300])
    col_vec = torch.tensor([[10], [20]])
    
    print(f"\n矩阵 + 行向量 [100, 200, 300]:\n{x + row_vec}")
    print(f"\n矩阵 + 列向量 [[10], [20]]:\n{x + col_vec}")


def physics_application_demo():
    """物理应用示例：电场计算"""
    print_section("8. 物理应用：点电荷电场")
    
    # 创建 2D 空间网格
    N = 50
    L = 5.0
    
    x = torch.linspace(-L, L, N)
    y = torch.linspace(-L, L, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 点电荷位置
    q = 1.0
    x0, y0 = 1.0, 0.0
    
    # 计算到电荷的距离
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    r = torch.clamp(r, min=0.1)  # 避免除零
    
    # 电势 V = kq/r (取 k=1)
    V = q / r
    
    print(f"空间网格: {N}x{N}")
    print(f"电荷位置: ({x0}, {y0})")
    print(f"电势范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"原点处电势: {V[N//2, N//2]:.4f}")
    
    # 计算电场（数值梯度）
    dx = x[1] - x[0]
    Ex, Ey = torch.gradient(V, spacing=(dx.item(), dx.item()))
    Ex, Ey = -Ex, -Ey  # E = -∇V
    
    E_magnitude = torch.sqrt(Ex**2 + Ey**2)
    print(f"电场强度范围: [{E_magnitude.min():.4f}, {E_magnitude.max():.4f}]")


def particle_simulation_demo():
    """物理应用示例：粒子距离矩阵"""
    print_section("9. 物理应用：多粒子系统")
    
    # 模拟 N 个粒子
    N = 100
    torch.manual_seed(42)
    
    # 粒子位置 [N, 3]
    positions = torch.randn(N, 3) * 2  # 3D 空间中随机分布
    
    print(f"粒子数: {N}")
    print(f"位置张量形状: {positions.shape}")
    
    # 使用广播计算两两距离
    # positions.unsqueeze(0): [1, N, 3]
    # positions.unsqueeze(1): [N, 1, 3]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 3]
    distances = torch.sqrt((diff ** 2).sum(dim=2))  # [N, N]
    
    print(f"距离矩阵形状: {distances.shape}")
    print(f"距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"平均距离: {distances.sum() / (N*(N-1)):.4f}")
    
    # 计算 Lennard-Jones 势能
    sigma, epsilon = 1.0, 1.0
    
    # 避免自相互作用
    distances_safe = distances + torch.eye(N) * 1e10
    
    r6 = (sigma / distances_safe) ** 6
    r12 = r6 ** 2
    V_LJ = 4 * epsilon * (r12 - r6)
    
    total_energy = V_LJ.sum() / 2  # 除以2避免重复计数
    print(f"Lennard-Jones 总势能: {total_energy:.4f}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print(" PyTorch 张量基础示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    
    # 运行所有示例
    basic_tensor_creation()
    data_types_demo()
    device_demo()
    numpy_interop_demo()
    tensor_operations_demo()
    linear_algebra_demo()
    broadcasting_demo()
    physics_application_demo()
    particle_simulation_demo()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

