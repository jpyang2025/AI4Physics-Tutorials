# 6.4 分布式训练

## 📖 概述

随着模型规模和数据量的增长，单 GPU 训练往往不够高效。分布式训练允许我们利用多个 GPU 或多台机器来加速训练过程。

## 🎯 学习目标

- 理解分布式训练的基本概念
- 掌握 DataParallel 和 DistributedDataParallel
- 了解混合精度训练
- 学会处理大规模训练的技巧

---

## 6.4.1 分布式训练概述

### 并行策略

```
分布式训练
│
├── 数据并行（Data Parallelism）
│   └── 每个 GPU 有完整模型副本，处理不同数据
│
├── 模型并行（Model Parallelism）
│   └── 模型太大，分布在多个 GPU 上
│
└── 流水线并行（Pipeline Parallelism）
    └── 将模型分成多个阶段，像流水线一样处理
```

### 物理类比

分布式训练类似于**并行计算**中的多处理器模拟：

| 概念 | 物理模拟类比 |
|------|-------------|
| 数据并行 | 空间分解法（每个处理器负责一部分空间） |
| 模型并行 | 任务分解法（不同处理器计算不同物理量） |
| 梯度同步 | 边界条件交换 |

---

## 6.4.2 DataParallel（单机多卡）

### 基本用法

```python
import torch
import torch.nn as nn

# 创建模型
model = YourModel()

# 检查可用 GPU
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU")
    model = nn.DataParallel(model)

model = model.cuda()

# 训练代码保持不变
for inputs, targets in train_loader:
    inputs = inputs.cuda()
    targets = targets.cuda()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### DataParallel 的工作原理

```
                         输入数据
                            │
                    ┌───────┴───────┐
                    ▼               ▼
               GPU 0 数据      GPU 1 数据
                    │               │
                    ▼               ▼
              模型副本 0       模型副本 1
                    │               │
                    ▼               ▼
                输出 0          输出 1
                    │               │
                    └───────┬───────┘
                            ▼
                    在 GPU 0 上汇总
                            │
                            ▼
                    计算损失和梯度
                            │
                            ▼
                    广播梯度到所有 GPU
                            │
                            ▼
                       更新参数
```

### DataParallel 的局限性

1. **GPU 0 成为瓶颈**：所有输出都汇集到 GPU 0
2. **显存不均衡**：GPU 0 占用更多显存
3. **Python GIL 限制**：多线程效率不高

---

## 6.4.3 DistributedDataParallel（推荐）

DistributedDataParallel (DDP) 是更高效的多 GPU 训练方式。

### 基本概念

```python
# 关键概念：
# - world_size: 总进程数（通常等于 GPU 数）
# - rank: 当前进程的全局 ID
# - local_rank: 当前进程在本机的 GPU ID
```

### 单机多卡 DDP

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPU 使用 nccl
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """训练函数（在每个进程中运行）"""
    
    # 初始化
    setup(rank, world_size)
    
    # 创建模型
    model = YourModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 创建数据加载器（使用分布式采样器）
    train_dataset = YourDataset()
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 重要：确保每个 epoch 的数据顺序不同
        
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 只在主进程打印和保存
        if rank == 0:
            print(f"Epoch {epoch} completed")
            torch.save(model.module.state_dict(), 'model.pth')
    
    cleanup()

def main():
    """主函数"""
    world_size = torch.cuda.device_count()
    
    # 启动多个进程
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

### 使用 torchrun 启动（推荐）

```python
# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 从环境变量获取分布式信息
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    # 初始化
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    # 创建模型
    model = YourModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    
    # ... 训练代码 ...
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

```bash
# 启动命令
# 单机 4 卡
torchrun --nproc_per_node=4 train_ddp.py

# 多机（2 机器，每机 4 卡）
# 机器 1
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" \
         --master_port=12355 --nproc_per_node=4 train_ddp.py

# 机器 2
torchrun --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" \
         --master_port=12355 --nproc_per_node=4 train_ddp.py
```

---

## 6.4.4 混合精度训练

### 使用 torch.cuda.amp

```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for inputs, targets in train_loader:
    inputs = inputs.cuda()
    targets = targets.cuda()
    
    optimizer.zero_grad()
    
    # 自动混合精度上下文
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 更新参数（自动处理梯度缩放）
    scaler.step(optimizer)
    scaler.update()
```

### DDP + 混合精度

```python
def train_ddp_amp(rank, world_size, args):
    """DDP + 混合精度训练"""
    
    setup(rank, world_size)
    
    model = YourModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    scaler = GradScaler()
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        for inputs, targets in train_loader:
            inputs = inputs.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    cleanup()
```

### 混合精度的物理意义

混合精度训练使用 FP16（16位浮点）进行前向/反向传播，FP32（32位）存储参数：

| 精度 | 数值范围 | 类比 |
|-----|---------|------|
| FP32 | ~1e-38 到 ~1e38 | 高精度科学计算 |
| FP16 | ~1e-8 到 ~65504 | 工程估算 |

关键在于**动态损失缩放**：防止小梯度在 FP16 中下溢。

---

## 6.4.5 梯度累积

当 GPU 显存不足以容纳大批量时使用。

```python
accumulation_steps = 4  # 累积 4 步等效于 4 倍 batch size
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.cuda()
    targets = targets.cuda()
    
    # 前向传播
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accumulation_steps  # 缩放损失
    
    # 反向传播（梯度累积）
    scaler.scale(loss).backward()
    
    # 每 accumulation_steps 步更新一次
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 6.4.6 梯度同步

### 同步 BatchNorm

```python
# 跨 GPU 同步 BatchNorm 统计量
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])
```

### 手动梯度同步

```python
# 在特定时刻同步梯度
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor /= world_size

# 广播参数
dist.broadcast(tensor, src=0)  # 从 rank 0 广播

# 收集所有进程的张量
gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered, tensor)
```

---

## 6.4.7 分布式训练实用工具

### 只在主进程执行

```python
def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def save_on_master(state, path):
    """只在主进程保存"""
    if is_main_process():
        torch.save(state, path)

def print_on_master(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
```

### 分布式采样器

```python
from torch.utils.data.distributed import DistributedSampler

# 创建分布式采样器
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,  # 总进程数
    rank=rank,                 # 当前进程
    shuffle=True,
    drop_last=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,    # 不要设置 shuffle=True
    num_workers=4,
    pin_memory=True
)

# 每个 epoch 开始时设置 epoch
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # 重要！
    # ... 训练 ...
```

---

## 6.4.8 完整 DDP 训练脚本

```python
#!/usr/bin/env python3
"""
完整的 DDP 训练脚本

使用方法：
    torchrun --nproc_per_node=4 train_ddp_complete.py
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        # 非分布式模式
        return 0, 1, 0
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    args = parse_args()
    
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process():
        print(f"训练配置: {world_size} GPUs, batch_size={args.batch_size}")
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    
    # 分布式采样器
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    # DDP 包装
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 训练
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        if is_main_process():
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # 保存模型
    if is_main_process():
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, 'model_ddp.pth')
        print("模型已保存")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
```

---

## 6.4.9 调试分布式训练

### 常见问题

```python
# 问题1：NCCL 超时
# 解决：增加超时时间
os.environ['NCCL_BLOCKING_WAIT'] = '1'
dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=2))

# 问题2：端口占用
# 解决：更换端口
os.environ['MASTER_PORT'] = '12356'

# 问题3：显存不均衡
# 解决：使用 DistributedDataParallel 而非 DataParallel

# 问题4：死锁
# 检查：确保所有进程执行相同的集合操作
```

### 调试技巧

```python
def debug_print(msg, rank=None):
    """带 rank 信息的调试打印"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] {msg}")

# 检查张量是否同步
def check_sync(tensor, name="tensor"):
    """检查张量在所有进程是否相同"""
    if not dist.is_initialized():
        return True
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    
    for i, t in enumerate(tensor_list[1:], 1):
        if not torch.allclose(tensor_list[0], t, atol=1e-6):
            print(f"警告：{name} 在 rank 0 和 rank {i} 不同步")
            return False
    return True
```

---

## 🔬 物理视角总结

### 并行计算的物理类比

| 分布式概念 | 物理模拟类比 |
|-----------|-------------|
| 数据并行 | 空间域分解 |
| 梯度同步 | 边界条件交换 |
| 批量大小 | 采样数/系综大小 |
| 混合精度 | 自适应精度积分 |

### 缩放定律

理想情况下，使用 $N$ 个 GPU 应该获得接近 $N$ 倍的加速：

$$T_N = \frac{T_1}{N} + T_{\text{comm}}$$

其中 $T_{\text{comm}}$ 是通信开销。

---

## 📝 练习

1. 将现有的单 GPU 训练脚本改写为 DDP 版本
2. 实现混合精度训练并比较速度提升
3. 尝试使用梯度累积模拟大批量训练

---

## ⏭️ 下一章预告

掌握了这些进阶技术后，第7章将展示如何将神经网络应用于物理学问题，包括求解微分方程、分子动力学模拟等。

