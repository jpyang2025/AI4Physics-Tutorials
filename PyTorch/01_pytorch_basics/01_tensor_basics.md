# 1.1 张量基础

## 什么是张量？

在物理学中，**张量**是一个在坐标变换下按照特定规则变换的多维数组。例如：
- **标量**（0阶张量）：温度、能量
- **矢量**（1阶张量）：速度、力
- **二阶张量**：应力张量、惯性张量
- **高阶张量**：弹性模量张量（4阶）

在 PyTorch 中，**Tensor** 是一种多维数组数据结构，类似于 NumPy 的 `ndarray`，但具有两个关键优势：
1. **GPU 加速**：可以在 GPU 上进行高速并行计算
2. **自动微分**：可以自动计算梯度，这是神经网络训练的核心

## 导入 PyTorch

```python
import torch
import numpy as np

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查是否有可用的 GPU
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

## 创建张量

### 1. 从 Python 列表创建

```python
# 一维张量（类似于物理中的矢量）
v = torch.tensor([1.0, 2.0, 3.0])
print(f"一维张量: {v}")
print(f"形状: {v.shape}")  # torch.Size([3])

# 二维张量（类似于矩阵或二阶张量）
M = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print(f"二维张量:\n{M}")
print(f"形状: {M.shape}")  # torch.Size([2, 3])

# 三维张量（例如：一批图像，或时空场）
T = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(f"三维张量形状: {T.shape}")  # torch.Size([2, 2, 2])
```

### 2. 使用工厂函数创建

```python
# 全零张量 - 类似于初始化一个场为零
zeros = torch.zeros(3, 4)
print(f"零张量:\n{zeros}")

# 全一张量
ones = torch.ones(2, 3)
print(f"全一张量:\n{ones}")

# 单位矩阵 - 在物理中非常常用
I = torch.eye(3)
print(f"单位矩阵:\n{I}")

# 随机张量 - 均匀分布 [0, 1)
rand_uniform = torch.rand(2, 3)
print(f"均匀分布随机张量:\n{rand_uniform}")

# 随机张量 - 标准正态分布 N(0, 1)
rand_normal = torch.randn(2, 3)
print(f"正态分布随机张量:\n{rand_normal}")

# 等差数列 - 类似于 np.linspace，常用于定义空间/时间网格
x = torch.linspace(0, 2*3.14159, 100)  # 0 到 2π，100个点
print(f"等差数列形状: {x.shape}")

# 类似于 np.arange
indices = torch.arange(0, 10, 2)  # 0, 2, 4, 6, 8
print(f"arange: {indices}")
```

### 3. 创建与现有张量形状相同的张量

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 创建形状相同的零张量
zeros_like_x = torch.zeros_like(x)

# 创建形状相同的随机张量
rand_like_x = torch.rand_like(x)

# 创建形状相同的全一张量
ones_like_x = torch.ones_like(x)

print(f"原张量:\n{x}")
print(f"相同形状的随机张量:\n{rand_like_x}")
```

## 数据类型 (dtype)

PyTorch 支持多种数据类型，选择合适的类型对计算效率和精度都很重要：

| dtype | 描述 | 典型用途 |
|-------|------|---------|
| `torch.float32` | 32位浮点（默认） | 神经网络权重、一般计算 |
| `torch.float64` | 64位浮点 | 需要高精度的科学计算 |
| `torch.float16` | 16位浮点 | GPU加速训练（混合精度） |
| `torch.int32` | 32位整数 | 索引、计数 |
| `torch.int64` | 64位整数 | 大范围索引 |
| `torch.bool` | 布尔值 | 掩码操作 |
| `torch.complex64` | 64位复数 | 量子力学、傅里叶变换 |
| `torch.complex128` | 128位复数 | 高精度复数计算 |

```python
# 指定数据类型
x_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
x_float64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
x_int = torch.tensor([1, 2, 3], dtype=torch.int64)

# 复数张量 - 物理中常用（如波函数）
psi = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(f"复数张量: {psi}")
print(f"实部: {psi.real}")
print(f"虚部: {psi.imag}")
print(f"模: {torch.abs(psi)}")

# 类型转换
x = torch.tensor([1.5, 2.7, 3.9])
x_int = x.to(torch.int32)  # 截断为整数
x_double = x.to(torch.float64)  # 转为双精度
x_double_alt = x.double()  # 等价写法
x_float = x_double.float()  # 转回单精度

print(f"原始: {x}")
print(f"转为整数: {x_int}")
```

## 设备 (Device): CPU vs GPU

PyTorch 可以在 CPU 或 GPU 上执行计算。GPU 并行计算能力强，适合大规模张量运算。

```python
# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 在 CPU 上创建张量
x_cpu = torch.randn(1000, 1000)
print(f"x_cpu 设备: {x_cpu.device}")  # cpu

# 将张量移动到 GPU（如果可用）
x_gpu = x_cpu.to(device)
print(f"x_gpu 设备: {x_gpu.device}")  # cuda:0 或 cpu

# 直接在 GPU 上创建张量
if torch.cuda.is_available():
    x_direct_gpu = torch.randn(1000, 1000, device="cuda")
    print(f"直接在 GPU 创建: {x_direct_gpu.device}")

# 注意：不同设备上的张量不能直接运算！
# y = x_cpu + x_gpu  # 这会报错！
# 需要先移到同一设备
y = x_cpu + x_gpu.cpu()  # 或 x_cpu.to(device) + x_gpu
```

### GPU 计算性能对比

```python
import time

size = 5000

# CPU 计算
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
z_cpu = torch.mm(x_cpu, y_cpu)  # 矩阵乘法
cpu_time = time.time() - start
print(f"CPU 矩阵乘法时间: {cpu_time:.4f} 秒")

# GPU 计算（如果可用）
if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()
    y_gpu = y_cpu.cuda()
    
    # 预热 GPU
    torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    z_gpu = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()  # 等待 GPU 完成
    gpu_time = time.time() - start
    print(f"GPU 矩阵乘法时间: {gpu_time:.4f} 秒")
    print(f"GPU 加速比: {cpu_time/gpu_time:.1f}x")
```

## 张量与 NumPy 的互转

PyTorch 与 NumPy 可以无缝互操作，这对于使用 SciPy、Matplotlib 等库非常有用。

### NumPy → PyTorch

```python
import numpy as np

# NumPy 数组
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])

# 方法1: torch.tensor() - 创建副本
t1 = torch.tensor(np_array)

# 方法2: torch.from_numpy() - 共享内存（修改一个会影响另一个）
t2 = torch.from_numpy(np_array)

print(f"NumPy 数组:\n{np_array}")
print(f"PyTorch 张量:\n{t1}")

# 共享内存演示
np_array[0, 0] = 999
print(f"修改 NumPy 后，torch.tensor(): {t1[0, 0]}")  # 仍然是 1.0
print(f"修改 NumPy 后，from_numpy(): {t2[0, 0]}")   # 变成 999
```

### PyTorch → NumPy

```python
# PyTorch 张量
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 转换为 NumPy（共享内存，CPU上）
np_from_tensor = tensor.numpy()

# 如果张量在 GPU 上，需要先移到 CPU
if torch.cuda.is_available():
    gpu_tensor = tensor.cuda()
    # np_array = gpu_tensor.numpy()  # 这会报错！
    np_array = gpu_tensor.cpu().numpy()  # 正确做法

# 如果张量需要梯度，需要先 detach()
tensor_with_grad = torch.tensor([1.0, 2.0], requires_grad=True)
# np_array = tensor_with_grad.numpy()  # 这会报错！
np_array = tensor_with_grad.detach().numpy()  # 正确做法
```

## 张量的属性

每个张量都有几个重要属性：

```python
x = torch.randn(3, 4, 5)

print(f"形状 (shape): {x.shape}")        # torch.Size([3, 4, 5])
print(f"维度数 (ndim): {x.ndim}")         # 3
print(f"总元素数 (numel): {x.numel()}")   # 60
print(f"数据类型 (dtype): {x.dtype}")     # torch.float32
print(f"设备 (device): {x.device}")       # cpu
print(f"是否需要梯度: {x.requires_grad}") # False
```

## 物理应用示例：电场的数值表示

让我们用张量来表示一个二维空间中的电场：

```python
import torch
import numpy as np

# 创建空间网格
N = 100  # 网格点数
L = 10.0  # 空间范围 [-L/2, L/2]

x = torch.linspace(-L/2, L/2, N)
y = torch.linspace(-L/2, L/2, N)
X, Y = torch.meshgrid(x, y, indexing='ij')

# 点电荷位置
q = 1.0  # 电荷量
x0, y0 = 2.0, 0.0  # 电荷位置

# 计算距离
r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
r = torch.clamp(r, min=0.1)  # 避免除零

# 电势 (V = kq/r, 取 k=1)
V = q / r

# 电场 E = -∇V（这里用数值差分近似）
dx = x[1] - x[0]
Ex = -torch.gradient(V, spacing=(dx,), dim=0)[0]  # PyTorch 2.0+
Ey = -torch.gradient(V, spacing=(dx,), dim=1)[0]

print(f"电势场形状: {V.shape}")
print(f"电场分量形状: Ex={Ex.shape}, Ey={Ey.shape}")
print(f"原点处电势: {V[N//2, N//2]:.4f}")
```

## 本节小结

| 概念 | 说明 |
|------|------|
| 张量创建 | `torch.tensor()`, `torch.zeros()`, `torch.randn()` 等 |
| 数据类型 | `float32`（默认）, `float64`, `complex64` 等 |
| 设备 | CPU（默认）或 CUDA（GPU） |
| NumPy互转 | `torch.from_numpy()`, `.numpy()` |

## 练习

1. 创建一个 $4 \times 4$ 的单位矩阵，将其对角元素改为 $[1, 2, 3, 4]$
2. 创建一个复数张量表示波函数 $\psi(x) = e^{ikx}$，其中 $k=2\pi$，$x \in [0, 1]$ 取 100 个点
3. 比较 `torch.tensor()` 和 `torch.from_numpy()` 创建张量时的内存占用差异

## 延伸阅读

- [PyTorch 官方张量教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch 张量 API 文档](https://pytorch.org/docs/stable/tensors.html)

---

[下一节：张量运算 →](./02_tensor_operations.md)

