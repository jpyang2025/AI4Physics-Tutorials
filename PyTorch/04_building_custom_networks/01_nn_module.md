# 4.1 nn.Module 详解

## nn.Module 是什么？

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。理解它的工作原理是构建自定义网络的基础。

一个 `nn.Module`：
- 可以包含**参数**（可学习的权重）
- 可以包含**子模块**（其他 nn.Module）
- 定义了**前向传播**逻辑
- 自动追踪参数和子模块

## 最简单的自定义模块

```python
import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    """最简单的自定义线性层"""
    
    def __init__(self, in_features, out_features):
        super().__init__()  # 必须调用父类构造函数
        
        # 定义可学习参数
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        """定义前向传播"""
        return x @ self.weight.T + self.bias

# 使用自定义模块
layer = SimpleLinear(10, 5)
x = torch.randn(3, 10)
y = layer(x)
print(f"输入形状: {x.shape}, 输出形状: {y.shape}")

# 查看参数
for name, param in layer.named_parameters():
    print(f"{name}: {param.shape}")
```

## nn.Module 的核心组成

### 1. `__init__` 方法

在 `__init__` 中定义网络的组件：

```python
class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 必须！初始化父类
        
        # 使用 nn.Parameter 定义可学习参数
        self.custom_param = nn.Parameter(torch.randn(hidden_dim))
        
        # 使用现有的层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # 使用容器
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # 注意：普通 Python 属性不会被追踪
        self.some_config = 42  # 这个不是参数
```

### 2. `forward` 方法

定义前向传播逻辑：

```python
class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """前向传播"""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 调用 forward 的正确方式
model = MyNetwork(10, 20, 5)
x = torch.randn(3, 10)
y = model(x)  # 不要直接调用 model.forward(x)！
```

**为什么用 `model(x)` 而不是 `model.forward(x)`？**

`model(x)` 会调用 `__call__` 方法，它除了调用 `forward` 还会：
- 执行注册的钩子（hooks）
- 处理 autograd 相关逻辑
- 进行其他必要的簿记操作

## nn.Parameter

`nn.Parameter` 是一个特殊的张量，会被自动注册为模块的参数：

```python
class ParameterDemo(nn.Module):
    def __init__(self):
        super().__init__()
        
        # nn.Parameter 会被注册为参数
        self.learnable = nn.Parameter(torch.randn(3, 3))
        
        # 普通张量不会被注册
        self.not_learnable = torch.randn(3, 3)
        
        # 如果需要不可学习但要保存的张量，使用 register_buffer
        self.register_buffer('constant', torch.ones(3, 3))
    
    def forward(self, x):
        return x @ self.learnable + self.constant

model = ParameterDemo()

print("注册的参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print("\n注册的缓冲区:")
for name, buf in model.named_buffers():
    print(f"  {name}: {buf.shape}")
```

### register_buffer 的用途

用于存储不需要梯度但需要随模型保存/加载的张量：

```python
class BatchNormManual(nn.Module):
    """手动实现的 BatchNorm（简化版）"""
    
    def __init__(self, num_features):
        super().__init__()
        
        # 可学习的缩放和偏移
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 运行统计量（不需要梯度，但需要保存）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # 更新运行统计量
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x_norm + self.beta
```

## 容器类

### nn.Sequential

顺序执行一系列层：

```python
# 方式1：按顺序传入
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 方式2：使用 OrderedDict 命名层
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))

# 可以通过名称访问
print(model.fc1)

# 前向传播
x = torch.randn(32, 784)
y = model(x)
```

### nn.ModuleList

当需要动态数量的层时使用：

```python
class DynamicMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        
        # 使用 ModuleList 存储可变数量的层
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        
        self.activations = nn.ModuleList([
            nn.ReLU() for _ in range(len(layer_sizes) - 2)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        return self.layers[-1](x)

# 不同深度的网络
shallow = DynamicMLP([10, 20, 5])
deep = DynamicMLP([10, 64, 64, 64, 64, 5])

print(f"浅网络层数: {len(shallow.layers)}")
print(f"深网络层数: {len(deep.layers)}")
```

**警告**：不要使用普通的 Python 列表！

```python
class WrongModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 错误！这些层不会被注册
        self.layers = [nn.Linear(10, 10) for _ in range(3)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

wrong_model = WrongModel()
print(f"参数数量: {sum(p.numel() for p in wrong_model.parameters())}")  # 0！
```

### nn.ModuleDict

使用字符串键访问子模块：

```python
class MultiHeadModel(nn.Module):
    """多任务模型：共享主干，不同的头"""
    
    def __init__(self, input_dim, hidden_dim, task_dims):
        super().__init__()
        
        # 共享主干
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 不同任务的头
        self.heads = nn.ModuleDict({
            task_name: nn.Linear(hidden_dim, output_dim)
            for task_name, output_dim in task_dims.items()
        })
    
    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)

# 创建多任务模型
model = MultiHeadModel(
    input_dim=100,
    hidden_dim=64,
    task_dims={'classification': 10, 'regression': 1, 'embedding': 32}
)

x = torch.randn(8, 100)
y_cls = model(x, 'classification')
y_reg = model(x, 'regression')
y_emb = model(x, 'embedding')

print(f"分类输出: {y_cls.shape}")
print(f"回归输出: {y_reg.shape}")
print(f"嵌入输出: {y_emb.shape}")
```

## 模块的常用方法

### 参数访问

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 获取所有参数
for param in model.parameters():
    print(f"形状: {param.shape}")

# 获取命名参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 统计参数量
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total}, 可训练: {trainable}")
```

### 子模块访问

```python
# 获取直接子模块
for name, module in model.named_children():
    print(f"{name}: {module}")

# 获取所有模块（递归）
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")
```

### 模式切换

```python
model.train()   # 训练模式
print(f"训练模式: {model.training}")

model.eval()    # 评估模式
print(f"训练模式: {model.training}")
```

### 设备转移

```python
model = model.to('cuda')      # 移到 GPU
model = model.to('cpu')       # 移回 CPU
model = model.cuda()          # 等价于 to('cuda')
model = model.cpu()           # 等价于 to('cpu')

# 数据类型转换
model = model.float()         # FP32
model = model.half()          # FP16
model = model.double()        # FP64
```

### 保存与加载

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()  # 先创建相同结构
model.load_state_dict(torch.load('model.pth'))
```

## 钩子 (Hooks)

钩子允许你在前向/反向传播时插入自定义操作：

### 前向钩子

```python
class HookDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = HookDemo()

# 存储中间激活
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 注册钩子
model.fc1.register_forward_hook(save_activation('fc1'))
model.fc2.register_forward_hook(save_activation('fc2'))

# 前向传播
x = torch.randn(2, 10)
y = model(x)

# 查看中间激活
for name, act in activations.items():
    print(f"{name}: {act.shape}")
```

### 反向钩子

```python
def gradient_hook(module, grad_input, grad_output):
    """监控梯度"""
    print(f"梯度形状: {grad_output[0].shape}")
    print(f"梯度范数: {grad_output[0].norm():.4f}")

model.fc1.register_full_backward_hook(gradient_hook)

x = torch.randn(2, 10)
y = model(x)
y.sum().backward()
```

## 自定义前向传播的灵活性

`forward` 方法可以非常灵活：

```python
class FlexibleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.decoder = nn.Linear(20, 10)
    
    def forward(self, x, return_latent=False):
        """可以返回不同的输出"""
        latent = self.encoder(x)
        
        if return_latent:
            return latent
        
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        """也可以定义其他方法"""
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

model = FlexibleModel()
x = torch.randn(4, 10)

# 不同的调用方式
recon = model(x)
latent = model(x, return_latent=True)
latent2 = model.encode(x)
```

## 物理应用：等变神经网络

以下是一个简单的平移等变网络示例：

```python
class TranslationEquivariantLayer(nn.Module):
    """
    平移等变层（1D 卷积）
    
    物理意义：如果输入平移，输出也相应平移
    f(T[x]) = T[f(x)]，其中 T 是平移操作
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 卷积天然具有平移等变性
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2  # same padding
        )
    
    def forward(self, x):
        return self.conv(x)

# 验证平移等变性
layer = TranslationEquivariantLayer(1, 1, 3)

x = torch.randn(1, 1, 10)
x_shifted = torch.roll(x, shifts=2, dims=2)

y = layer(x)
y_shifted = layer(x_shifted)

# y_shifted 应该约等于 torch.roll(y, shifts=2, dims=2)
y_expected = torch.roll(y, shifts=2, dims=2)

print(f"等变性误差: {(y_shifted - y_expected).abs().mean():.6f}")
```

## 本节小结

| 概念 | 说明 |
|------|------|
| `nn.Module` | 所有网络模块的基类 |
| `__init__` | 定义网络组件 |
| `forward` | 定义前向传播 |
| `nn.Parameter` | 可学习参数 |
| `register_buffer` | 不可学习但需保存的张量 |
| `nn.Sequential` | 顺序容器 |
| `nn.ModuleList` | 模块列表（动态层数） |
| `nn.ModuleDict` | 模块字典（按名访问） |

## 练习

1. 实现一个自定义的 Dropout 层，使用 `nn.Parameter` 和 `register_buffer`
2. 创建一个支持可变深度的残差网络，使用 `nn.ModuleList`
3. 使用前向钩子可视化 CNN 中间层的特征图

## 延伸阅读

- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Hooks 教程](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html)

---

[返回章节目录](./README.md) | [下一节：常用层类型 →](./02_layer_types.md)

