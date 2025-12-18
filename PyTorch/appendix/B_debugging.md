# 附录 B：调试技巧

## 📖 概述

本附录介绍 PyTorch 开发中常见问题的诊断和调试方法，帮助你快速定位和解决问题。

---

## B.1 常见错误及解决方案

### 形状不匹配

**错误信息**：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x64 and 128x10)
```

**调试方法**：

```python
def debug_shapes(model, x):
    """打印每层的输入输出形状"""
    print(f"输入形状: {x.shape}")
    
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name}: {x.shape}")
    
    return x


# 使用钩子函数
def register_hooks(model):
    """注册钩子打印中间形状"""
    def hook(module, input, output):
        print(f"{module.__class__.__name__}: {input[0].shape} -> {output.shape}")
    
    for layer in model.children():
        layer.register_forward_hook(hook)


# 逐层检查
x = torch.randn(32, 3, 224, 224)
print(f"输入: {x.shape}")

x = model.conv1(x)
print(f"conv1 后: {x.shape}")

x = model.pool(x)
print(f"pool 后: {x.shape}")

# 找到问题层
x = x.view(x.size(0), -1)
print(f"展平后: {x.shape}")
print(f"fc1 期望输入: {model.fc1.in_features}")
```

### 设备不匹配

**错误信息**：
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**调试方法**：

```python
def check_device(model, data):
    """检查设备一致性"""
    # 模型设备
    model_device = next(model.parameters()).device
    print(f"模型设备: {model_device}")
    
    # 数据设备
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"数据 '{k}' 设备: {v.device}")
    else:
        print(f"数据设备: {data.device}")
    
    # 检查一致性
    if isinstance(data, torch.Tensor):
        assert data.device == model_device, "设备不匹配！"


# 确保一致性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

### 梯度问题

**错误信息**：
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**调试方法**：

```python
def check_gradients(model):
    """检查梯度状态"""
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  grad: {param.grad is not None}")
        if param.grad is not None:
            print(f"  grad norm: {param.grad.norm().item():.6f}")


# 检查计算图
def check_computation_graph(tensor):
    """检查张量的计算图"""
    print(f"requires_grad: {tensor.requires_grad}")
    print(f"grad_fn: {tensor.grad_fn}")
    print(f"is_leaf: {tensor.is_leaf}")


# 确保梯度流动
x = torch.randn(10, requires_grad=True)
y = model(x)
print(f"输出 requires_grad: {y.requires_grad}")

# 检查是否有梯度
loss = criterion(y, target)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"警告: {name} 没有梯度！")
    elif param.grad.abs().sum() == 0:
        print(f"警告: {name} 梯度为零！")
```

---

## B.2 调试工具

### 使用 print 调试

```python
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        print(f"[DEBUG] 输入: {x.shape}, device={x.device}")
        
        x = self.conv(x)
        print(f"[DEBUG] conv 后: {x.shape}")
        
        x = F.adaptive_avg_pool2d(x, 1)
        print(f"[DEBUG] pool 后: {x.shape}")
        
        x = x.view(x.size(0), -1)
        print(f"[DEBUG] flatten 后: {x.shape}")
        
        x = self.fc(x)
        print(f"[DEBUG] fc 后: {x.shape}")
        
        return x
```

### 使用 PyTorch hooks

```python
class ActivationLogger:
    """记录激活值用于调试"""
    
    def __init__(self, model):
        self.activations = {}
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input': input[0].detach() if input else None,
                    'output': output.detach() if isinstance(output, torch.Tensor) else None
                }
            return hook
        
        for name, layer in model.named_modules():
            if name:
                layer.register_forward_hook(get_hook(name))
    
    def print_stats(self):
        for name, act in self.activations.items():
            if act['output'] is not None:
                out = act['output']
                print(f"{name}:")
                print(f"  shape: {out.shape}")
                print(f"  mean: {out.mean():.4f}, std: {out.std():.4f}")
                print(f"  min: {out.min():.4f}, max: {out.max():.4f}")


# 使用
logger = ActivationLogger(model)
output = model(input)
logger.print_stats()
```

### 使用 torch.autograd.detect_anomaly

```python
# 检测梯度异常
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # 如果有 NaN/Inf 会抛出详细错误
```

### 使用 torch.autograd.gradcheck

```python
# 检查自定义函数的梯度是否正确
from torch.autograd import gradcheck

def my_function(x):
    return x ** 2 + 2 * x

x = torch.randn(10, requires_grad=True, dtype=torch.float64)
result = gradcheck(my_function, x, eps=1e-6, atol=1e-4, rtol=1e-3)
print(f"梯度检查通过: {result}")
```

---

## B.3 内存调试

### 显存监控

```python
def print_gpu_memory():
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 显存: 已分配 {allocated:.2f} GB, 已缓存 {cached:.2f} GB")


def gpu_memory_tracker(func):
    """显存追踪装饰器"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"峰值显存: {peak:.2f} GB")
        
        return result
    return wrapper


# 使用
@gpu_memory_tracker
def train_step(model, inputs, targets):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss
```

### 显存泄漏检测

```python
def detect_memory_leak(model, dataloader, n_iterations=10):
    """检测显存泄漏"""
    print("开始检测显存泄漏...")
    
    memories = []
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= n_iterations:
            break
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        memories.append(torch.cuda.memory_allocated())
        
        print(f"Iteration {i}: {memories[-1] / 1024**2:.2f} MB")
    
    # 检查是否有持续增长
    if memories[-1] > memories[0] * 1.5:
        print("⚠️ 警告: 可能存在显存泄漏！")
    else:
        print("✓ 显存使用正常")
```

### 常见显存泄漏原因

```python
# ❌ 错误：在循环中累积张量
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # 保留了计算图！

# ✓ 正确：使用 .item() 或 .detach()
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # 只保留数值


# ❌ 错误：没有 torch.no_grad() 进行评估
def evaluate(model, dataloader):
    model.eval()
    for batch in dataloader:
        output = model(batch)  # 仍然在构建计算图！

# ✓ 正确
def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
```

---

## B.4 数值稳定性

### 检测 NaN/Inf

```python
def check_tensor_health(tensor, name="tensor"):
    """检查张量是否包含 NaN 或 Inf"""
    if torch.isnan(tensor).any():
        print(f"⚠️ {name} 包含 NaN！")
        return False
    if torch.isinf(tensor).any():
        print(f"⚠️ {name} 包含 Inf！")
        return False
    return True


def check_model_health(model):
    """检查模型参数健康状况"""
    for name, param in model.named_parameters():
        if not check_tensor_health(param, f"参数 {name}"):
            return False
        if param.grad is not None:
            if not check_tensor_health(param.grad, f"梯度 {name}"):
                return False
    return True


# 在训练中使用
for epoch in range(epochs):
    for batch in dataloader:
        loss = train_step(batch)
        
        if not check_model_health(model):
            print(f"模型在 epoch {epoch} 出现数值问题！")
            break
```

### 梯度裁剪

```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# 监控梯度范数
def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

print(f"梯度范数: {get_gradient_norm(model):.4f}")
```

---

## B.5 性能调试

### 识别瓶颈

```python
import time

class Timer:
    """简单计时器"""
    
    def __init__(self, name=""):
        self.name = name
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start
        print(f"{self.name}: {self.elapsed*1000:.2f} ms")


# 使用
with Timer("数据加载"):
    batch = next(iter(dataloader))

with Timer("前向传播"):
    output = model(input)

with Timer("反向传播"):
    loss.backward()

with Timer("优化器步骤"):
    optimizer.step()
```

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(input)

# 打印结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Chrome trace
prof.export_chrome_trace("trace.json")
```

### 数据加载性能

```python
def benchmark_dataloader(dataloader, n_batches=100):
    """测试 DataLoader 性能"""
    start = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
    
    elapsed = time.time() - start
    print(f"加载 {n_batches} 批次用时: {elapsed:.2f}s")
    print(f"每批次: {elapsed/n_batches*1000:.2f}ms")
    
    # 建议
    batch_time = elapsed / n_batches
    if batch_time > 0.1:  # 超过 100ms
        print("建议: 考虑增加 num_workers")


# 测试不同 num_workers
for num_workers in [0, 2, 4, 8]:
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
    print(f"\nnum_workers={num_workers}:")
    benchmark_dataloader(loader)
```

---

## B.6 调试技巧汇总

### 调试清单

1. **形状问题**
   - [ ] 打印每一步的张量形状
   - [ ] 检查 batch 维度是否正确
   - [ ] 验证卷积层的输出尺寸

2. **设备问题**
   - [ ] 确认模型在正确设备上
   - [ ] 确认所有输入数据在同一设备
   - [ ] 使用 `.to(device)` 统一管理

3. **梯度问题**
   - [ ] 检查 `requires_grad` 设置
   - [ ] 验证损失函数是否可微
   - [ ] 检查是否有 `detach()` 切断梯度

4. **内存问题**
   - [ ] 使用 `torch.no_grad()` 进行评估
   - [ ] 使用 `.item()` 获取标量值
   - [ ] 定期清理 GPU 缓存

5. **数值问题**
   - [ ] 检查是否有 NaN/Inf
   - [ ] 使用梯度裁剪
   - [ ] 降低学习率

### 快速调试命令

```python
# 快速形状检查
print({name: p.shape for name, p in model.named_parameters()})

# 快速梯度检查
print({name: p.grad is not None for name, p in model.named_parameters()})

# 快速设备检查
print(next(model.parameters()).device)

# 快速内存检查
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

---

## B.7 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 损失为 NaN | 学习率过大/数值溢出 | 降低学习率，使用梯度裁剪 |
| 损失不下降 | 学习率过小/模型问题 | 增大学习率，检查模型结构 |
| 显存爆炸 | batch size 过大 | 减小 batch size，使用梯度累积 |
| 训练速度慢 | 数据加载瓶颈 | 增加 num_workers，使用 pin_memory |
| 精度不提升 | 过拟合/欠拟合 | 调整模型容量，添加正则化 |

