# 附录 C：最佳实践

## 📖 概述

本附录总结了 PyTorch 开发中的最佳实践，帮助你编写更高效、更可维护的代码。

---

## C.1 代码组织

### 项目结构

```
project/
├── config/
│   ├── __init__.py
│   ├── default.yaml          # 默认配置
│   └── experiment.yaml       # 实验配置
├── data/
│   ├── __init__.py
│   ├── dataset.py            # 数据集定义
│   ├── transforms.py         # 数据转换
│   └── dataloader.py         # 数据加载器
├── models/
│   ├── __init__.py
│   ├── base.py               # 基础模型类
│   ├── layers.py             # 自定义层
│   └── networks.py           # 网络架构
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py       # 基础训练器
│   └── trainer.py            # 具体训练逻辑
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # 评估指标
│   ├── visualization.py      # 可视化
│   └── checkpoint.py         # 检查点管理
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── inference.py          # 推理脚本
├── notebooks/
│   └── analysis.ipynb        # 分析笔记本
├── tests/
│   └── test_model.py         # 单元测试
├── requirements.txt
└── README.md
```

### 模型定义模板

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    """模型描述
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: dropout 比例
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 保存超参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 定义层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())
```

### 训练器模板

```python
class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # 状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs: int):
        """完整训练"""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pt')
            
            self.current_epoch += 1
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filename)
```

---

## C.2 性能优化

### 数据加载优化

```python
# ✓ 推荐的 DataLoader 配置
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # 多进程加载
    pin_memory=True,         # 固定内存加速传输
    prefetch_factor=2,       # 预取批次数
    persistent_workers=True  # 保持 worker 进程
)

# 对于小数据集，可以预加载到 GPU
class PreloadedDataset:
    def __init__(self, dataset, device):
        self.data = []
        for x, y in dataset:
            self.data.append((x.to(device), y.to(device)))
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
```

### GPU 优化

```python
# 启用 cuDNN benchmark（输入尺寸固定时）
torch.backends.cudnn.benchmark = True

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


# 使用 torch.compile（PyTorch 2.0+）
model = torch.compile(model)
```

### 内存优化

```python
# 梯度累积（模拟大 batch size）
accumulation_steps = 4

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()


# 梯度检查点（用时间换内存）
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return self.output(x)


# 及时释放不需要的张量
del intermediate_tensor
torch.cuda.empty_cache()
```

---

## C.3 可复现性

### 设置随机种子

```python
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性算法（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在训练开始时调用
set_seed(42)
```

### 配置管理

```python
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class TrainConfig:
    # 数据
    data_path: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    
    # 模型
    model_name: str = "resnet18"
    hidden_dim: int = 256
    dropout: float = 0.1
    
    # 训练
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # 其他
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)


# 使用
config = TrainConfig(lr=1e-4, epochs=50)
config.to_yaml('config.yaml')
```

---

## C.4 实验管理

### 日志记录

```python
import logging
from datetime import datetime


def setup_logging(log_dir: str):
    """设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging("./logs")
logger.info(f"开始训练，配置: {config}")
```

### TensorBoard 集成

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# 记录标量
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)

# 记录超参数
writer.add_hparams(
    {'lr': config.lr, 'batch_size': config.batch_size},
    {'final_accuracy': best_accuracy}
)

# 记录模型结构
writer.add_graph(model, sample_input)

# 记录图像
writer.add_images('predictions', images, epoch)

# 记录直方图
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

writer.close()
```

### 实验跟踪框架

```python
# 使用 Weights & Biases
import wandb

wandb.init(
    project="my-project",
    config={
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32
    }
)

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

wandb.finish()
```

---

## C.5 模型验证

### 单元测试

```python
import unittest
import torch


class TestModel(unittest.TestCase):
    
    def setUp(self):
        self.model = MyModel(input_dim=10, output_dim=5)
        self.model.eval()
    
    def test_forward_shape(self):
        """测试输出形状"""
        x = torch.randn(32, 10)
        y = self.model(x)
        self.assertEqual(y.shape, (32, 5))
    
    def test_forward_batch_sizes(self):
        """测试不同 batch size"""
        for batch_size in [1, 16, 64]:
            x = torch.randn(batch_size, 10)
            y = self.model(x)
            self.assertEqual(y.shape[0], batch_size)
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        self.model.train()
        x = torch.randn(4, 10, requires_grad=True)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 没有梯度")
    
    def test_save_load(self):
        """测试保存和加载"""
        x = torch.randn(4, 10)
        y1 = self.model(x)
        
        # 保存和加载
        torch.save(self.model.state_dict(), 'test_model.pt')
        new_model = MyModel(input_dim=10, output_dim=5)
        new_model.load_state_dict(torch.load('test_model.pt'))
        new_model.eval()
        
        y2 = new_model(x)
        self.assertTrue(torch.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()
```

### 过拟合测试

```python
def overfit_single_batch(model, batch, epochs=100):
    """测试模型能否过拟合单个批次"""
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            acc = (outputs.argmax(1) == targets).float().mean()
            print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")
    
    # 最终应该接近 100% 准确率
    final_acc = (model(inputs).argmax(1) == targets).float().mean()
    print(f"最终准确率: {final_acc.item():.4f}")
    
    if final_acc > 0.99:
        print("✓ 模型可以正确学习")
    else:
        print("⚠️ 模型可能有问题")
```

---

## C.6 部署准备

### 模型导出

```python
# TorchScript 导出
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# 或使用 trace
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')


# ONNX 导出
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
```

### 推理优化

```python
# 使用 torch.inference_mode
@torch.inference_mode()
def predict(model, inputs):
    return model(inputs)


# 批量推理
def batch_predict(model, data, batch_size=32):
    model.eval()
    results = []
    
    with torch.inference_mode():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            output = model(batch)
            results.append(output)
    
    return torch.cat(results)


# 使用 torch.compile
compiled_model = torch.compile(model, mode='reduce-overhead')
```

---

## C.7 检查清单

### 训练前

- [ ] 设置随机种子
- [ ] 验证数据加载正确
- [ ] 测试模型可以过拟合小数据
- [ ] 检查损失函数计算正确
- [ ] 验证梯度正常流动

### 训练中

- [ ] 监控损失和指标
- [ ] 定期保存检查点
- [ ] 检查梯度范数
- [ ] 监控 GPU 使用率

### 训练后

- [ ] 评估测试集性能
- [ ] 可视化结果
- [ ] 保存最佳模型
- [ ] 记录实验配置

### 部署前

- [ ] 导出模型（TorchScript/ONNX）
- [ ] 验证导出模型正确性
- [ ] 测试推理性能
- [ ] 编写推理文档

