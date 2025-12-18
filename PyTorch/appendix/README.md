# 附录

## 📖 概述

本附录提供了 PyTorch 开发的实用参考资料，包括环境配置、调试技巧、最佳实践和学习资源。

## 📚 附录内容

| 附录 | 文件 | 主题 |
|------|------|------|
| A | [A_installation.md](./A_installation.md) | 安装与环境配置 |
| B | [B_debugging.md](./B_debugging.md) | 调试技巧 |
| C | [C_best_practices.md](./C_best_practices.md) | 最佳实践 |
| D | [D_resources.md](./D_resources.md) | 资源推荐 |

## 🔧 快速参考

### 常用命令

```bash
# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU 信息
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 清除 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 常用导入

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
```

### 设备管理

```python
# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据移动到设备
model = model.to(device)
inputs = inputs.to(device)
```

### 常见形状变换

```python
# 展平
x.view(-1)              # 完全展平
x.view(batch_size, -1)  # 保留 batch 维度

# 增加/删除维度
x.unsqueeze(0)   # 在第 0 维增加
x.squeeze()      # 删除所有大小为 1 的维度

# 维度重排
x.permute(0, 2, 1)  # 交换维度
x.transpose(1, 2)   # 交换两个维度
```

## 📊 速查表

### 激活函数

| 函数 | 公式 | 适用场景 |
|------|------|---------|
| ReLU | max(0, x) | 隐藏层默认选择 |
| LeakyReLU | max(0.01x, x) | 避免死亡神经元 |
| GELU | x·Φ(x) | Transformer |
| Sigmoid | 1/(1+e⁻ˣ) | 二分类输出 |
| Softmax | eˣⁱ/Σeˣʲ | 多分类输出 |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNN、PINN |

### 损失函数

| 任务 | 损失函数 | PyTorch 类 |
|------|---------|-----------|
| 回归 | MSE | `nn.MSELoss()` |
| 回归 | MAE | `nn.L1Loss()` |
| 二分类 | BCE | `nn.BCEWithLogitsLoss()` |
| 多分类 | 交叉熵 | `nn.CrossEntropyLoss()` |
| 分割 | Dice Loss | 自定义 |

### 优化器

| 优化器 | 典型学习率 | 特点 |
|--------|-----------|------|
| SGD | 0.01-0.1 | 简单，需要调参 |
| SGD+Momentum | 0.01-0.1 | 加速收敛 |
| Adam | 1e-4-1e-3 | 自适应，易用 |
| AdamW | 1e-4-1e-3 | 更好的正则化 |

### 数据类型

| dtype | 说明 | 用途 |
|-------|------|------|
| torch.float32 | 32位浮点 | 默认/训练 |
| torch.float16 | 16位浮点 | 混合精度 |
| torch.int64 | 64位整数 | 索引/标签 |
| torch.bool | 布尔值 | 掩码 |

## ⚠️ 常见错误速查

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `CUDA out of memory` | GPU 显存不足 | 减小 batch size，使用混合精度 |
| `Expected ... but got ...` | 张量类型/形状不匹配 | 检查数据类型和形状 |
| `RuntimeError: element 0 of tensors...` | 标量需要梯度 | 使用 `loss.item()` 获取值 |
| `Trying to backward through the graph a second time` | 计算图被释放 | 使用 `retain_graph=True` |
| `one of the variables needed for gradient computation has been modified` | 原地操作问题 | 避免对需要梯度的张量进行原地操作 |

## 🔗 快捷链接

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [torchvision 模型](https://pytorch.org/vision/stable/models.html)

---

*附录持续更新中*

