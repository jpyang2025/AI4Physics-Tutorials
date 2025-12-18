# 6.3 模型保存与加载

## 📖 概述

正确保存和加载模型是深度学习工作流程中的重要环节。本节介绍如何保存模型权重、完整模型、训练检查点，以及如何导出模型用于部署。

## 🎯 学习目标

- 掌握保存和加载模型的不同方式
- 理解 state_dict 的结构
- 实现完整的检查点管理
- 了解模型导出（ONNX、TorchScript）

---

## 6.3.1 保存和加载 state_dict

### 什么是 state_dict？

`state_dict` 是一个 Python 字典，将每个层映射到其参数张量。

```python
import torch
import torch.nn as nn

# 创建简单模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 查看 state_dict
print("模型的 state_dict:")
for name, param in model.state_dict().items():
    print(f"  {name}: {param.shape}")

# 输出：
# 模型的 state_dict:
#   0.weight: torch.Size([20, 10])
#   0.bias: torch.Size([20])
#   2.weight: torch.Size([5, 20])
#   2.bias: torch.Size([5])
```

### 推荐方式：只保存 state_dict

```python
# 保存
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 切换到评估模式
```

### 加载到不同设备

```python
# 保存（在 GPU 上训练）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载到 CPU
model.load_state_dict(
    torch.load('model_weights.pth', map_location='cpu')
)

# 加载到指定 GPU
model.load_state_dict(
    torch.load('model_weights.pth', map_location='cuda:0')
)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load('model_weights.pth', map_location=device)
)
model.to(device)
```

---

## 6.3.2 保存完整模型

### 使用 pickle 保存整个模型

```python
# 保存整个模型（包括结构和权重）
torch.save(model, 'complete_model.pth')

# 加载
model = torch.load('complete_model.pth')
model.eval()
```

### ⚠️ 注意事项

**不推荐保存完整模型**，因为：

1. **依赖序列化**：模型类的定义必须存在于加载环境中
2. **可移植性差**：不同 PyTorch 版本可能不兼容
3. **文件更大**：包含了模型结构信息

```python
# ❌ 可能出问题的情况
# 保存时
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

model = MyModel()
torch.save(model, 'model.pth')

# 加载时，如果 MyModel 类的定义不存在或已更改
model = torch.load('model.pth')  # 可能失败！
```

---

## 6.3.3 检查点（Checkpoint）

### 完整检查点包含的内容

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """保存完整检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    torch.save(checkpoint, path)
    print(f"检查点已保存: {path}")


def load_checkpoint(path, model, optimizer, scheduler=None, device='cpu'):
    """加载检查点"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复随机状态（确保可重复性）
    if checkpoint.get('rng_state') is not None:
        torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"从 epoch {epoch} 恢复训练，损失: {loss:.4f}")
    
    return epoch, loss
```

### 使用检查点恢复训练

```python
import os

def train_with_checkpoint(model, train_loader, val_loader, criterion, 
                          optimizer, scheduler, num_epochs, 
                          checkpoint_dir='checkpoints', resume_from=None):
    """支持检查点恢复的训练"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 尝试恢复
    if resume_from and os.path.exists(resume_from):
        start_epoch, _ = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        start_epoch += 1  # 从下一个 epoch 开始
    
    device = next(model.parameters()).device
    
    for epoch in range(start_epoch, num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(checkpoint_dir, 'best_model.pth')
            )
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )
```

---

## 6.3.4 处理不匹配的 state_dict

### 部分加载

```python
def load_partial_state_dict(model, state_dict, strict=False):
    """
    部分加载 state_dict
    
    用于迁移学习或模型结构变化的情况
    """
    model_state = model.state_dict()
    
    # 过滤出匹配的键
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                pretrained_dict[k] = v
            else:
                print(f"跳过 {k}: 形状不匹配 {v.shape} vs {model_state[k].shape}")
        else:
            print(f"跳过 {k}: 模型中不存在")
    
    # 更新 state_dict
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    
    print(f"加载了 {len(pretrained_dict)}/{len(state_dict)} 个参数")


# 使用
pretrained_state = torch.load('pretrained_model.pth')
load_partial_state_dict(model, pretrained_state)
```

### 重命名键

```python
def rename_state_dict_keys(state_dict, key_mapping):
    """
    重命名 state_dict 中的键
    
    Args:
        state_dict: 原始 state_dict
        key_mapping: {old_key: new_key} 的字典
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = key_mapping.get(k, k)
        new_state_dict[new_key] = v
    return new_state_dict


# 示例：从旧模型迁移到新模型
key_mapping = {
    'fc1.weight': 'encoder.fc1.weight',
    'fc1.bias': 'encoder.fc1.bias',
    'fc2.weight': 'decoder.fc2.weight',
    'fc2.bias': 'decoder.fc2.bias',
}

old_state = torch.load('old_model.pth')
new_state = rename_state_dict_keys(old_state, key_mapping)
model.load_state_dict(new_state)
```

---

## 6.3.5 DataParallel 模型的保存与加载

```python
# DataParallel 会给所有键添加 'module.' 前缀

# 保存 DataParallel 模型
model = nn.DataParallel(model)
torch.save(model.module.state_dict(), 'model.pth')  # 保存 .module

# 或者保存整个 state_dict，加载时处理前缀
torch.save(model.state_dict(), 'model_dp.pth')

# 加载时去除 'module.' 前缀
def remove_module_prefix(state_dict):
    """移除 DataParallel 的 'module.' 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除前7个字符
        else:
            new_state_dict[k] = v
    return new_state_dict


state_dict = torch.load('model_dp.pth')
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
```

---

## 6.3.6 TorchScript 导出

TorchScript 可以将 PyTorch 模型序列化为可独立运行的格式。

### Tracing

```python
import torch.jit

# 准备示例输入
example_input = torch.randn(1, 3, 224, 224)

# 跟踪模型
model.eval()
traced_model = torch.jit.trace(model, example_input)

# 保存
traced_model.save('model_traced.pt')

# 加载
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
```

### Scripting

```python
# 对于包含控制流的模型，使用 script
class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x, use_relu=True):
        x = self.fc1(x)
        if use_relu:
            x = torch.relu(x)
        return self.fc2(x)

model = ConditionalModel()
scripted_model = torch.jit.script(model)

scripted_model.save('model_scripted.pt')
```

### TorchScript 的优势

1. **独立于 Python**：可以在 C++ 环境中运行
2. **优化**：JIT 编译器可以进行优化
3. **可移植**：无需原始模型定义
4. **生产部署**：适合服务端部署

---

## 6.3.7 ONNX 导出

ONNX（Open Neural Network Exchange）是一个开放的模型格式。

```python
import torch.onnx

# 准备模型和示例输入
model.eval()
example_input = torch.randn(1, 3, 224, 224)

# 导出到 ONNX
torch.onnx.export(
    model,                          # 模型
    example_input,                  # 示例输入
    'model.onnx',                   # 输出文件
    export_params=True,             # 导出参数
    opset_version=11,               # ONNX 算子版本
    do_constant_folding=True,       # 常量折叠优化
    input_names=['input'],          # 输入名称
    output_names=['output'],        # 输出名称
    dynamic_axes={                  # 动态轴（支持可变 batch size）
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("模型已导出为 ONNX 格式")
```

### 验证 ONNX 模型

```python
import onnx
import onnxruntime as ort
import numpy as np

# 验证模型
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过")

# 使用 ONNX Runtime 推理
ort_session = ort.InferenceSession('model.onnx')

# 准备输入
input_name = ort_session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 推理
output = ort_session.run(None, {input_name: input_data})
print(f"输出形状: {output[0].shape}")
```

---

## 6.3.8 检查点管理器

```python
import os
import glob
from datetime import datetime

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir, max_to_keep=5):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, epoch, metrics, 
             is_best=False, filename=None):
        """保存检查点"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 生成文件名
        if filename is None:
            filename = f'checkpoint_epoch_{epoch:04d}.pth'
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"保存检查点: {path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"更新最佳模型: {best_path}")
        
        # 清理旧检查点
        self._cleanup()
        
        return path
    
    def load(self, path, model, optimizer=None, scheduler=None, device='cpu'):
        """加载检查点"""
        
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"加载检查点: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
        
        return checkpoint
    
    def load_best(self, model, optimizer=None, scheduler=None, device='cpu'):
        """加载最佳模型"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
            return self.load(best_path, model, optimizer, scheduler, device)
        else:
            raise FileNotFoundError("最佳模型不存在")
    
    def load_latest(self, model, optimizer=None, scheduler=None, device='cpu'):
        """加载最新检查点"""
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        )
        if not checkpoints:
            raise FileNotFoundError("没有找到检查点")
        
        latest = max(checkpoints, key=os.path.getctime)
        return self.load(latest, model, optimizer, scheduler, device)
    
    def _cleanup(self):
        """清理旧检查点，只保留最新的 max_to_keep 个"""
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        )
        
        if len(checkpoints) > self.max_to_keep:
            # 按修改时间排序
            checkpoints.sort(key=os.path.getctime)
            
            # 删除最旧的
            for ckpt in checkpoints[:-self.max_to_keep]:
                os.remove(ckpt)
                print(f"删除旧检查点: {ckpt}")
    
    def list_checkpoints(self):
        """列出所有检查点"""
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, '*.pth')
        )
        for ckpt in sorted(checkpoints):
            info = torch.load(ckpt, map_location='cpu')
            print(f"{os.path.basename(ckpt)}:")
            print(f"  Epoch: {info.get('epoch', 'N/A')}")
            print(f"  Timestamp: {info.get('timestamp', 'N/A')}")


# 使用示例
manager = CheckpointManager('checkpoints', max_to_keep=5)

# 训练中保存
for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
    
    manager.save(
        model, optimizer, scheduler, epoch,
        metrics={'train_loss': train_loss, 'val_loss': val_loss},
        is_best=is_best
    )

# 恢复训练
checkpoint = manager.load_latest(model, optimizer, scheduler)
start_epoch = checkpoint['epoch'] + 1
```

---

## 6.3.9 安全保存（防止损坏）

```python
import shutil

def safe_save(obj, path):
    """
    安全保存，防止在保存过程中中断导致文件损坏
    """
    temp_path = path + '.tmp'
    
    # 先保存到临时文件
    torch.save(obj, temp_path)
    
    # 如果成功，替换原文件
    shutil.move(temp_path, path)


def save_with_backup(obj, path):
    """
    保存并保留备份
    """
    if os.path.exists(path):
        backup_path = path + '.backup'
        shutil.copy(path, backup_path)
    
    safe_save(obj, path)
```

---

## 🔬 物理视角总结

### 检查点的意义

检查点可以类比于物理模拟中的**状态快照**：

- 保存系统的完整状态（位置、速度等）
- 可以从任意时刻恢复模拟
- 便于分析中间状态

### 模型导出

将模型导出（TorchScript、ONNX）类似于：

- 将数值模拟代码编译为可执行程序
- 可以在不同平台运行
- 优化后运行更快

---

## 📝 练习

1. 实现一个完整的检查点管理系统
2. 将训练好的模型导出为 ONNX 格式并验证
3. 实现从预训练模型部分加载权重的功能

---

## ⏭️ 下一节

下一节我们将学习 [分布式训练](./04_distributed_training.md)，了解如何在多 GPU 上训练模型。

