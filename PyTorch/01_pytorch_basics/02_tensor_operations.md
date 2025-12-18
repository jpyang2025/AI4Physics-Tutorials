# 1.2 张量运算

## 概述

张量运算是科学计算的核心。本节将介绍 PyTorch 中常用的张量操作，包括索引切片、形状变换、数学运算和广播机制。如果你熟悉 NumPy，会发现 PyTorch 的语法非常相似。

## 索引与切片

PyTorch 的索引语法与 NumPy 完全一致：

### 基本索引

```python
import torch

# 创建示例张量
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(f"原张量:\n{x}")
print(f"形状: {x.shape}")  # [3, 4]

# 单个元素
print(f"x[0, 0] = {x[0, 0]}")  # 1
print(f"x[1, 2] = {x[1, 2]}")  # 7

# 整行/整列
print(f"第一行: {x[0]}")      # [1, 2, 3, 4]
print(f"第一行: {x[0, :]}")   # 等价写法
print(f"第一列: {x[:, 0]}")   # [1, 5, 9]

# 切片 [start:end:step]
print(f"前两行: \n{x[:2]}")
print(f"后两列: \n{x[:, 2:]}")
print(f"隔行取: \n{x[::2]}")  # 第0行和第2行
```

### 高级索引

```python
# 布尔索引 - 非常有用的筛选操作
x = torch.randn(5, 4)
print(f"原张量:\n{x}")

# 找出所有正数
mask = x > 0
positives = x[mask]
print(f"正数: {positives}")

# 将负数设为0（类似ReLU激活函数）
x_relu = x.clone()
x_relu[x_relu < 0] = 0
print(f"ReLU 后:\n{x_relu}")

# 花式索引
indices = torch.tensor([0, 2, 4])
print(f"选取第0、2、4行:\n{x[indices]}")

# 多维花式索引
row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([1, 2, 3])
print(f"选取 (0,1), (1,2), (2,3): {x[row_idx, col_idx]}")
```

### 物理应用：选取特定能量范围的粒子

```python
# 模拟粒子数据：[N_particles, features]
# features: [px, py, pz, E, charge]
particles = torch.randn(1000, 5)
particles[:, 4] = torch.randint(-1, 2, (1000,)).float()  # 电荷 -1, 0, 1

# 计算动量大小
p = torch.sqrt(particles[:, 0]**2 + particles[:, 1]**2 + particles[:, 2]**2)

# 选取动量在某范围内的粒子
p_min, p_max = 0.5, 1.5
mask = (p > p_min) & (p < p_max)
selected = particles[mask]
print(f"动量在 [{p_min}, {p_max}] 范围内的粒子数: {selected.shape[0]}")

# 选取正电荷粒子
positive_particles = particles[particles[:, 4] > 0]
print(f"正电荷粒子数: {positive_particles.shape[0]}")
```

## 形状变换

### reshape 和 view

```python
x = torch.arange(12)
print(f"原张量: {x}")
print(f"形状: {x.shape}")  # [12]

# reshape: 改变形状
x_2d = x.reshape(3, 4)
print(f"reshape(3, 4):\n{x_2d}")

x_3d = x.reshape(2, 2, 3)
print(f"reshape(2, 2, 3):\n{x_3d}")

# 使用 -1 自动推断维度
x_auto = x.reshape(4, -1)  # 4行，自动计算列数
print(f"reshape(4, -1) 形状: {x_auto.shape}")  # [4, 3]

# view: 功能类似，但要求内存连续
x_view = x.view(3, 4)
print(f"view(3, 4):\n{x_view}")

# 注意：view 和 reshape 的区别
# - view 要求张量内存连续，否则报错
# - reshape 总是有效，必要时会复制数据
```

### squeeze 和 unsqueeze

这两个操作在处理批次维度时非常常用：

```python
# squeeze: 移除大小为1的维度
x = torch.randn(1, 3, 1, 4)
print(f"原形状: {x.shape}")  # [1, 3, 1, 4]

x_squeezed = x.squeeze()
print(f"squeeze() 后: {x_squeezed.shape}")  # [3, 4]

x_squeeze_dim = x.squeeze(0)  # 只移除第0维
print(f"squeeze(0) 后: {x_squeeze_dim.shape}")  # [3, 1, 4]

# unsqueeze: 添加大小为1的维度
y = torch.randn(3, 4)
print(f"原形状: {y.shape}")  # [3, 4]

y_unsqueeze = y.unsqueeze(0)  # 在第0维添加
print(f"unsqueeze(0) 后: {y_unsqueeze.shape}")  # [1, 3, 4]

y_unsqueeze2 = y.unsqueeze(-1)  # 在最后添加
print(f"unsqueeze(-1) 后: {y_unsqueeze2.shape}")  # [3, 4, 1]
```

### transpose 和 permute

```python
# transpose: 交换两个维度
x = torch.randn(2, 3, 4)
print(f"原形状: {x.shape}")  # [2, 3, 4]

x_t = x.transpose(0, 2)  # 交换第0和第2维
print(f"transpose(0, 2) 后: {x_t.shape}")  # [4, 3, 2]

# 2D 矩阵转置的便捷方法
matrix = torch.randn(3, 4)
print(f"矩阵形状: {matrix.shape}")
print(f"转置后: {matrix.T.shape}")  # [4, 3]

# permute: 任意重排维度
x = torch.randn(2, 3, 4, 5)
x_permuted = x.permute(3, 0, 2, 1)  # 按指定顺序重排
print(f"permute(3, 0, 2, 1) 后: {x_permuted.shape}")  # [5, 2, 4, 3]
```

### flatten 和 unflatten

```python
# flatten: 将多维张量展平
x = torch.randn(2, 3, 4)
print(f"原形状: {x.shape}")

x_flat = x.flatten()  # 完全展平
print(f"flatten() 后: {x_flat.shape}")  # [24]

x_flat_partial = x.flatten(start_dim=1)  # 从第1维开始展平
print(f"flatten(start_dim=1) 后: {x_flat_partial.shape}")  # [2, 12]

# 这在神经网络中常用：将卷积层输出展平后送入全连接层
```

## 数学运算

### 逐元素运算

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 2.0, 2.0, 2.0])

# 四则运算
print(f"加法: {x + y}")
print(f"减法: {x - y}")
print(f"乘法: {x * y}")  # 逐元素乘法！
print(f"除法: {x / y}")
print(f"幂运算: {x ** y}")

# 等价的函数形式
print(f"torch.add: {torch.add(x, y)}")
print(f"torch.mul: {torch.mul(x, y)}")

# 常用数学函数
print(f"平方根: {torch.sqrt(x)}")
print(f"指数: {torch.exp(x)}")
print(f"对数: {torch.log(x)}")
print(f"绝对值: {torch.abs(x - 2.5)}")
print(f"正弦: {torch.sin(x)}")
print(f"余弦: {torch.cos(x)}")

# 取整操作
z = torch.tensor([1.4, 2.5, 3.6, -1.5])
print(f"向下取整: {torch.floor(z)}")
print(f"向上取整: {torch.ceil(z)}")
print(f"四舍五入: {torch.round(z)}")
print(f"截断: {torch.trunc(z)}")
```

### 聚合运算

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# 全局聚合
print(f"求和: {x.sum()}")
print(f"均值: {x.mean()}")
print(f"最大值: {x.max()}")
print(f"最小值: {x.min()}")
print(f"标准差: {x.std()}")

# 沿指定维度聚合
print(f"按行求和 (dim=1): {x.sum(dim=1)}")  # [6, 15]
print(f"按列求和 (dim=0): {x.sum(dim=0)}")  # [5, 7, 9]

# 保持维度（常用于广播）
print(f"按行求和，保持维度: {x.sum(dim=1, keepdim=True)}")

# argmax/argmin: 返回最值的索引
print(f"最大值索引: {x.argmax()}")
print(f"按列最大值索引: {x.argmax(dim=0)}")
```

### 矩阵运算

```python
# 矩阵乘法 - 在物理中无处不在
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# 矩阵乘法的三种等价写法
C1 = torch.mm(A, B)        # 专用于2D矩阵
C2 = torch.matmul(A, B)    # 更通用，支持批量和广播
C3 = A @ B                 # Python 3.5+ 运算符

print(f"矩阵乘法结果形状: {C1.shape}")  # [3, 5]

# 批量矩阵乘法 (Batch Matrix Multiplication)
# 常用于同时处理多个样本
batch_A = torch.randn(10, 3, 4)  # 10个 3x4 矩阵
batch_B = torch.randn(10, 4, 5)  # 10个 4x5 矩阵
batch_C = torch.bmm(batch_A, batch_B)
print(f"批量矩阵乘法结果形状: {batch_C.shape}")  # [10, 3, 5]

# 或使用 @ 运算符（支持广播）
batch_C2 = batch_A @ batch_B
print(f"使用 @ 运算符: {batch_C2.shape}")

# 矩阵-向量乘法
M = torch.randn(3, 4)
v = torch.randn(4)
result = M @ v  # 或 torch.mv(M, v)
print(f"矩阵-向量乘法结果形状: {result.shape}")  # [3]

# 向量内积
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(a, b)
print(f"向量内积: {dot_product}")  # 32.0

# 向量外积
outer = torch.outer(a, b)
print(f"向量外积:\n{outer}")
```

### 线性代数运算

```python
# 创建一个正定矩阵（确保可逆）
A = torch.randn(4, 4)
A = A @ A.T + torch.eye(4)  # 正定矩阵

# 行列式
det = torch.linalg.det(A)
print(f"行列式: {det}")

# 矩阵求逆
A_inv = torch.linalg.inv(A)
print(f"A @ A_inv ≈ I:\n{A @ A_inv}")

# 矩阵的迹
trace = torch.trace(A)
print(f"迹: {trace}")

# 特征值和特征向量
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"特征值: {eigenvalues}")

# 对于实对称矩阵，使用 eigh 更高效
sym_A = (A + A.T) / 2  # 确保对称
eigenvalues_real, eigenvectors_real = torch.linalg.eigh(sym_A)
print(f"实特征值: {eigenvalues_real}")

# 奇异值分解 (SVD)
M = torch.randn(3, 5)
U, S, Vh = torch.linalg.svd(M)
print(f"SVD: U={U.shape}, S={S.shape}, Vh={Vh.shape}")

# 求解线性方程组 Ax = b
b = torch.randn(4)
x = torch.linalg.solve(A, b)
print(f"解 x 的形状: {x.shape}")
print(f"验证 Ax - b ≈ 0: {torch.allclose(A @ x, b)}")

# 矩阵范数
norm_fro = torch.linalg.norm(A, ord='fro')  # Frobenius 范数
norm_2 = torch.linalg.norm(A, ord=2)        # 谱范数
print(f"Frobenius 范数: {norm_fro}")
print(f"谱范数: {norm_2}")
```

## 广播机制 (Broadcasting)

广播是一种强大的机制，允许不同形状的张量进行运算：

```python
# 广播的基本规则：
# 1. 从最右边的维度开始比较
# 2. 维度相等，或其中一个为1，则可以广播
# 3. 缺失的维度视为1

# 示例1：标量与张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
scalar = 10
print(f"张量 + 标量:\n{x + scalar}")

# 示例2：向量与矩阵
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])  # 形状 [2, 3]
row_vec = torch.tensor([10, 20, 30])  # 形状 [3]
col_vec = torch.tensor([[100], [200]])  # 形状 [2, 1]

print(f"矩阵形状: {matrix.shape}")
print(f"行向量形状: {row_vec.shape}")
print(f"列向量形状: {col_vec.shape}")

# 行向量广播到每一行
print(f"矩阵 + 行向量:\n{matrix + row_vec}")

# 列向量广播到每一列
print(f"矩阵 + 列向量:\n{matrix + col_vec}")

# 示例3：创建网格（常用于场的计算）
x = torch.linspace(0, 1, 5)  # [5]
y = torch.linspace(0, 1, 4)  # [4]

# 使用广播创建 2D 函数值
# x.unsqueeze(1): [5, 1], y.unsqueeze(0): [1, 4]
# 广播后都变成 [5, 4]
z = torch.sin(x.unsqueeze(1) * 3.14159) * torch.cos(y.unsqueeze(0) * 3.14159)
print(f"2D 函数值形状: {z.shape}")  # [5, 4]
```

### 物理应用：多体相互作用

```python
# 计算 N 个粒子之间的两两距离
N = 100
positions = torch.randn(N, 3)  # N 个粒子的 3D 位置

# 使用广播计算距离矩阵
# positions: [N, 3]
# positions.unsqueeze(0): [1, N, 3]
# positions.unsqueeze(1): [N, 1, 3]
# 广播后相减: [N, N, 3]
diff = positions.unsqueeze(0) - positions.unsqueeze(1)
distances = torch.sqrt((diff ** 2).sum(dim=2))

print(f"距离矩阵形状: {distances.shape}")  # [N, N]
print(f"对角线应该为0: {distances.diag()[:5]}")  # 验证

# 计算 Lennard-Jones 势能
sigma = 1.0
epsilon = 1.0

# 避免自相互作用（对角线）
distances_safe = distances + torch.eye(N) * 1e10

# LJ 势: 4ε[(σ/r)^12 - (σ/r)^6]
r6 = (sigma / distances_safe) ** 6
r12 = r6 ** 2
V_LJ = 4 * epsilon * (r12 - r6)

# 总势能（除以2避免重复计数）
total_energy = V_LJ.sum() / 2
print(f"Lennard-Jones 总势能: {total_energy:.4f}")
```

## 张量拼接与分割

```python
# 拼接 (concatenate)
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 沿行方向拼接
cat_row = torch.cat([a, b], dim=0)
print(f"按行拼接:\n{cat_row}")

# 沿列方向拼接
cat_col = torch.cat([a, b], dim=1)
print(f"按列拼接:\n{cat_col}")

# stack: 沿新维度拼接
stacked = torch.stack([a, b], dim=0)
print(f"堆叠后形状: {stacked.shape}")  # [2, 2, 2]

# 分割
x = torch.arange(12).reshape(4, 3)
print(f"原张量:\n{x}")

# 均匀分割
chunks = torch.chunk(x, 2, dim=0)  # 分成2块
print(f"chunk 分成 {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"  块 {i}: 形状 {chunk.shape}")

# 按指定大小分割
splits = torch.split(x, [1, 3], dim=0)  # 分成大小为1和3的两块
print(f"split 分成 {len(splits)} 块")
for i, split in enumerate(splits):
    print(f"  块 {i}: 形状 {split.shape}")
```

## 原地操作 (In-place Operations)

PyTorch 支持原地操作，可以节省内存，但使用时需谨慎：

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(f"原始: {x}")

# 原地操作以 _ 结尾
x.add_(10)  # 等价于 x = x + 10
print(f"原地加法后: {x}")

x.mul_(2)   # 等价于 x = x * 2
print(f"原地乘法后: {x}")

x.zero_()   # 将所有元素设为0
print(f"原地清零后: {x}")

# 注意：需要梯度的张量进行原地操作可能导致问题
# 因为这会破坏计算图
x = torch.tensor([1.0, 2.0], requires_grad=True)
# x.add_(1)  # 这会报错或导致问题！
```

## 本节小结

| 操作类别 | 常用函数 |
|---------|---------|
| 索引切片 | `[]`, 布尔索引, `torch.where()` |
| 形状变换 | `reshape()`, `view()`, `squeeze()`, `unsqueeze()`, `transpose()`, `permute()` |
| 数学运算 | `+`, `-`, `*`, `/`, `@`, `torch.mm()`, `torch.exp()`, `torch.sin()` |
| 聚合运算 | `sum()`, `mean()`, `max()`, `min()`, `argmax()` |
| 线性代数 | `torch.linalg.inv()`, `torch.linalg.eig()`, `torch.linalg.solve()` |
| 拼接分割 | `torch.cat()`, `torch.stack()`, `torch.chunk()`, `torch.split()` |

## 练习

1. 创建一个 $5 \times 5$ 的矩阵，计算其特征值和特征向量，验证 $Av = \lambda v$
2. 使用广播机制计算两组点之间的欧氏距离矩阵（$M$ 个点和 $N$ 个点）
3. 实现归一化操作：将矩阵的每一行归一化到 $[0, 1]$ 范围

## 延伸阅读

- [PyTorch 张量操作文档](https://pytorch.org/docs/stable/torch.html)
- [PyTorch 线性代数文档](https://pytorch.org/docs/stable/linalg.html)

---

[← 上一节：张量基础](./01_tensor_basics.md) | [下一节：自动微分 →](./03_autograd.md)

