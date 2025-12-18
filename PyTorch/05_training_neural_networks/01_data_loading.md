# 5.1 数据加载

## 📖 概述

高效的数据加载是训练神经网络的基础。PyTorch 提供了 `Dataset` 和 `DataLoader` 两个核心类，实现了数据访问与批量处理的解耦。

## 🎯 学习目标

- 理解 `Dataset` 抽象基类
- 使用 `DataLoader` 进行批量数据加载
- 实现自定义数据集
- 掌握数据增强技术

---

## 5.1.1 Dataset 基础

### Dataset 抽象类

`torch.utils.data.Dataset` 是所有数据集的抽象基类。自定义数据集需要实现两个方法：

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """自定义数据集模板"""
    
    def __init__(self, data, labels):
        """
        初始化数据集
        
        Args:
            data: 输入数据
            labels: 标签数据
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            元组 (数据, 标签)
        """
        return self.data[idx], self.labels[idx]
```

### 物理数据集示例：谐振子轨迹

```python
import torch
import numpy as np
from torch.utils.data import Dataset

class HarmonicOscillatorDataset(Dataset):
    """
    谐振子轨迹数据集
    
    生成简谐运动的 (t, x) 数据对
    x(t) = A * cos(ωt + φ)
    """
    
    def __init__(self, n_trajectories=1000, n_points=100, 
                 omega_range=(0.5, 2.0), noise_level=0.1):
        """
        Args:
            n_trajectories: 轨迹数量
            n_points: 每条轨迹的时间点数
            omega_range: 角频率范围
            noise_level: 噪声水平
        """
        self.n_trajectories = n_trajectories
        self.n_points = n_points
        
        # 时间网格
        self.t = torch.linspace(0, 10, n_points)
        
        # 生成数据
        self.trajectories = []
        self.parameters = []  # (A, omega, phi)
        
        for _ in range(n_trajectories):
            # 随机参数
            A = np.random.uniform(0.5, 2.0)
            omega = np.random.uniform(*omega_range)
            phi = np.random.uniform(0, 2 * np.pi)
            
            # 生成轨迹
            x = A * torch.cos(omega * self.t + phi)
            
            # 添加噪声
            x = x + noise_level * torch.randn_like(x)
            
            self.trajectories.append(x)
            self.parameters.append(torch.tensor([A, omega, phi]))
        
        self.trajectories = torch.stack(self.trajectories)
        self.parameters = torch.stack(self.parameters)
    
    def __len__(self):
        return self.n_trajectories
    
    def __getitem__(self, idx):
        """返回 (轨迹, 参数)"""
        return self.trajectories[idx], self.parameters[idx]


# 使用示例
dataset = HarmonicOscillatorDataset(n_trajectories=1000)
print(f"数据集大小: {len(dataset)}")

# 获取一个样本
trajectory, params = dataset[0]
print(f"轨迹形状: {trajectory.shape}")
print(f"参数 (A, ω, φ): {params}")
```

---

## 5.1.2 DataLoader

`DataLoader` 将 `Dataset` 包装为可迭代对象，提供批量化、打乱、并行加载等功能。

### 基本用法

```python
from torch.utils.data import DataLoader

# 创建数据集
dataset = HarmonicOscillatorDataset(n_trajectories=1000)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,      # 批量大小
    shuffle=True,       # 打乱数据（训练时使用）
    num_workers=4,      # 并行加载的工作进程数
    pin_memory=True,    # 使用锁页内存（GPU 训练时加速）
    drop_last=True      # 丢弃最后不完整的批次
)

# 迭代数据
for batch_trajectories, batch_params in dataloader:
    print(f"批量轨迹形状: {batch_trajectories.shape}")  # [32, 100]
    print(f"批量参数形状: {batch_params.shape}")        # [32, 3]
    break
```

### DataLoader 关键参数

| 参数 | 说明 | 典型值 |
|------|------|-------|
| `batch_size` | 每批样本数 | 32, 64, 128, 256 |
| `shuffle` | 是否打乱 | 训练 True，测试 False |
| `num_workers` | 并行进程数 | 0-8（根据 CPU 核心数） |
| `pin_memory` | 锁页内存 | GPU 训练时 True |
| `drop_last` | 丢弃不完整批次 | 批归一化时 True |

### 批量大小的物理意义

批量大小影响梯度估计的方差：

$$\text{Var}[\nabla L_{\text{batch}}] \propto \frac{\sigma^2}{B}$$

其中 $B$ 是批量大小。

| 批量大小 | 特点 |
|---------|------|
| 小批量（8-32） | 噪声大，有助于跳出局部极小，但训练不稳定 |
| 中等批量（64-256） | 平衡噪声和稳定性，最常用 |
| 大批量（512+） | 梯度估计准确，需要更大学习率 |

---

## 5.1.3 内置数据集

PyTorch 提供了多种预置数据集，方便快速实验。

### torchvision 数据集

```python
import torchvision
import torchvision.transforms as transforms

# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 下载并加载 MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data',          # 数据存放路径
    train=True,             # 训练集
    transform=transform,    # 应用变换
    download=True           # 自动下载
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 查看数据
images, labels = next(iter(train_loader))
print(f"图像形状: {images.shape}")  # [64, 1, 28, 28]
print(f"标签形状: {labels.shape}")  # [64]
```

### 常用数据集一览

```python
# 图像分类
torchvision.datasets.MNIST       # 手写数字
torchvision.datasets.CIFAR10     # 10类自然图像
torchvision.datasets.CIFAR100    # 100类自然图像
torchvision.datasets.ImageNet    # 大规模图像分类
torchvision.datasets.FashionMNIST  # 服装图像

# 目标检测
torchvision.datasets.CocoDetection  # COCO 数据集

# 语义分割
torchvision.datasets.VOCSegmentation  # Pascal VOC
```

---

## 5.1.4 自定义数据集

### 从文件加载数据

```python
import os
import numpy as np
from torch.utils.data import Dataset

class ExperimentalDataset(Dataset):
    """
    从 .npy 文件加载实验数据
    
    适用于物理实验数据，如：
    - 光谱数据
    - 散射数据
    - 时间序列测量
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: 数据目录，包含 data.npy 和 labels.npy
            transform: 可选的数据变换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载数据
        self.data = np.load(os.path.join(data_dir, 'data.npy'))
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        
        # 转换为张量
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class LazyLoadDataset(Dataset):
    """
    懒加载数据集 - 适用于大型数据
    
    不将所有数据加载到内存，而是在需要时读取
    """
    
    def __init__(self, file_list, load_func):
        """
        Args:
            file_list: 数据文件路径列表
            load_func: 加载单个文件的函数
        """
        self.file_list = file_list
        self.load_func = load_func
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 仅在需要时加载数据
        return self.load_func(self.file_list[idx])
```

### 映射式与可迭代式数据集

```python
from torch.utils.data import Dataset, IterableDataset

# 映射式数据集（支持随机访问）
class MapDataset(Dataset):
    """通过索引访问"""
    def __getitem__(self, idx):
        return self.data[idx]

# 可迭代式数据集（流式访问）
class StreamDataset(IterableDataset):
    """
    适用于：
    - 实时数据流（传感器数据）
    - 超大型数据集
    - 在线生成的数据
    """
    
    def __init__(self, generator_func):
        self.generator_func = generator_func
    
    def __iter__(self):
        return self.generator_func()


# 示例：生成粒子散射事件
def particle_event_generator():
    """模拟粒子散射事件生成器"""
    while True:
        # 模拟入射粒子
        energy = np.random.exponential(10.0)  # GeV
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # 模拟出射粒子
        n_particles = np.random.poisson(5)
        particles = {
            'energy': energy,
            'theta': theta,
            'phi': phi,
            'n_out': n_particles
        }
        
        yield particles


stream_dataset = StreamDataset(particle_event_generator)
stream_loader = DataLoader(stream_dataset, batch_size=32)
```

---

## 5.1.5 数据变换与增强

### 基本变换

```python
import torchvision.transforms as T

# 图像变换管道
image_transform = T.Compose([
    T.Resize(256),              # 调整大小
    T.CenterCrop(224),          # 中心裁剪
    T.ToTensor(),               # 转换为张量 [0, 1]
    T.Normalize(                # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 数据增强（训练时使用）

数据增强可以有效扩充数据集，提高模型泛化能力。

```python
train_transform = T.Compose([
    T.RandomResizedCrop(224),           # 随机裁剪
    T.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
    T.RandomRotation(15),               # 随机旋转
    T.ColorJitter(                       # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 测试时不使用数据增强
test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
```

### 物理数据增强

```python
class PhysicsDataAugmentation:
    """物理数据增强器"""
    
    def __init__(self, noise_level=0.05, shift_range=0.1):
        self.noise_level = noise_level
        self.shift_range = shift_range
    
    def add_noise(self, x):
        """添加高斯噪声 - 模拟测量误差"""
        noise = torch.randn_like(x) * self.noise_level * x.abs().mean()
        return x + noise
    
    def time_shift(self, x):
        """时间平移 - 对于时间序列"""
        shift = int(np.random.uniform(-self.shift_range, self.shift_range) * len(x))
        return torch.roll(x, shifts=shift)
    
    def scale(self, x, scale_range=(0.8, 1.2)):
        """缩放 - 模拟增益变化"""
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    def __call__(self, x):
        """随机应用增强"""
        if np.random.random() > 0.5:
            x = self.add_noise(x)
        if np.random.random() > 0.5:
            x = self.time_shift(x)
        if np.random.random() > 0.5:
            x = self.scale(x)
        return x
```

---

## 5.1.6 数据划分

### 训练/验证/测试划分

```python
from torch.utils.data import random_split

# 假设有一个完整数据集
full_dataset = HarmonicOscillatorDataset(n_trajectories=10000)

# 划分数据集
train_size = int(0.8 * len(full_dataset))  # 80% 训练
val_size = int(0.1 * len(full_dataset))    # 10% 验证
test_size = len(full_dataset) - train_size - val_size  # 10% 测试

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子
)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")
print(f"测试集: {len(test_dataset)}")

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### K 折交叉验证

```python
from sklearn.model_selection import KFold

def create_kfold_loaders(dataset, k=5, batch_size=32):
    """创建 K 折交叉验证的数据加载器"""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    
    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        # 创建子集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
        print(f"Fold {fold+1}: Train {len(train_idx)}, Val {len(val_idx)}")
    
    return fold_loaders
```

---

## 5.1.7 自定义采样器

### 采样器类型

```python
from torch.utils.data import Sampler, SequentialSampler, RandomSampler, WeightedRandomSampler

# 顺序采样
seq_sampler = SequentialSampler(dataset)

# 随机采样
rand_sampler = RandomSampler(dataset)

# 带权重随机采样 - 处理类别不平衡
# 假设有两个类别，0类有900个样本，1类有100个样本
class_counts = [900, 100]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]  # labels 是每个样本的类别

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=weighted_sampler)
```

### 物理应用：稀有事件采样

```python
class RareEventSampler(Sampler):
    """
    稀有事件采样器
    
    在物理实验中，某些事件（如希格斯玻色子衰变）非常稀有。
    该采样器增加对稀有事件的采样概率。
    """
    
    def __init__(self, event_types, rare_event_ids, rare_boost=10.0):
        """
        Args:
            event_types: 事件类型数组
            rare_event_ids: 稀有事件的类型 ID 列表
            rare_boost: 稀有事件采样权重提升倍数
        """
        self.n_samples = len(event_types)
        
        # 计算采样权重
        self.weights = torch.ones(self.n_samples)
        for rare_id in rare_event_ids:
            rare_mask = (event_types == rare_id)
            self.weights[rare_mask] = rare_boost
        
        # 归一化
        self.weights = self.weights / self.weights.sum()
    
    def __iter__(self):
        indices = torch.multinomial(
            self.weights, 
            self.n_samples, 
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.n_samples
```

---

## 5.1.8 collate_fn 自定义批处理

当数据样本大小不一致时，需要自定义批处理函数。

```python
def custom_collate_fn(batch):
    """
    自定义批处理函数
    
    处理变长序列的批处理
    """
    # batch 是 [(data, label), (data, label), ...] 的列表
    data_list, label_list = zip(*batch)
    
    # 获取最长序列长度
    max_len = max(d.shape[0] for d in data_list)
    
    # 填充到相同长度
    padded_data = []
    lengths = []
    for d in data_list:
        length = d.shape[0]
        lengths.append(length)
        
        if length < max_len:
            padding = torch.zeros(max_len - length, *d.shape[1:])
            d = torch.cat([d, padding], dim=0)
        padded_data.append(d)
    
    return (
        torch.stack(padded_data),       # [B, max_len, ...]
        torch.tensor(lengths),           # [B]
        torch.stack(label_list)          # [B, ...]
    )


# 使用自定义 collate_fn
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)
```

### 变长粒子事件处理

```python
def particle_event_collate(batch):
    """
    粒子物理事件批处理
    
    每个事件有不同数量的粒子
    """
    events = []
    event_lengths = []
    targets = []
    
    for event_particles, target in batch:
        events.append(event_particles)
        event_lengths.append(len(event_particles))
        targets.append(target)
    
    # 找到最大粒子数
    max_particles = max(event_lengths)
    particle_dim = events[0].shape[-1]  # 每个粒子的特征维度
    
    # 填充
    padded_events = torch.zeros(len(batch), max_particles, particle_dim)
    mask = torch.zeros(len(batch), max_particles, dtype=torch.bool)
    
    for i, (event, length) in enumerate(zip(events, event_lengths)):
        padded_events[i, :length] = event
        mask[i, :length] = True
    
    return {
        'particles': padded_events,     # [B, max_n, D]
        'mask': mask,                    # [B, max_n]
        'n_particles': torch.tensor(event_lengths),  # [B]
        'target': torch.stack(targets)  # [B, ...]
    }
```

---

## 5.1.9 多 GPU 数据加载

```python
from torch.utils.data import DistributedSampler

# 分布式训练时使用
distributed_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # GPU 数量
    rank=rank,                # 当前 GPU 编号
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=distributed_sampler,  # 使用分布式采样器
    num_workers=4
)

# 每个 epoch 开始时需要设置 epoch
for epoch in range(num_epochs):
    distributed_sampler.set_epoch(epoch)
    for batch in loader:
        # 训练...
        pass
```

---

## 🔬 物理视角总结

### 数据加载的物理类比

| 数据加载概念 | 物理类比 |
|-------------|---------|
| Dataset | 相空间中的样本集合 |
| DataLoader | 遍历相空间的算法 |
| Batch | 系综中的子系统 |
| Shuffle | 遍历性假设 |
| 数据增强 | 对称性约束下的变换 |
| 采样权重 | 玻尔兹曼因子 |

### 批量大小与梯度估计

从统计力学角度，小批量梯度可以看作有限温度下的力：

$$F = -\nabla U + \sqrt{2T}\eta$$

其中温度 $T \propto 1/B$（B 是批量大小），$\eta$ 是随机噪声。

---

## 📝 练习

1. 为你的实验数据创建一个自定义 Dataset
2. 实现一个数据增强管道，包含噪声添加和缩放
3. 使用 WeightedRandomSampler 处理类别不平衡数据

---

## ⏭️ 下一节

下一节我们将学习 [优化器](./02_optimizers.md)，了解如何更新网络参数。

