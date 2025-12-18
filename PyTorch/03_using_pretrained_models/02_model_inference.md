# 3.2 模型推理

## 什么是推理？

**推理**（Inference）是指使用训练好的模型对新数据进行预测的过程。与训练不同，推理：

- 不需要计算梯度
- 不更新模型参数
- 通常需要更快的速度
- 可能需要部署到不同环境

## 推理模式 vs 训练模式

PyTorch 模型有两种模式，某些层（如 Dropout、BatchNorm）在两种模式下行为不同：

```python
import torch
import torchvision.models as models

model = models.resnet18(weights='IMAGENET1K_V1')

# 训练模式（默认）
model.train()
print(f"训练模式: {model.training}")  # True

# 评估/推理模式
model.eval()
print(f"评估模式: {model.training}")  # False
```

### 为什么要切换模式？

| 层类型 | 训练模式 | 评估模式 |
|--------|---------|---------|
| Dropout | 随机丢弃神经元 | 不丢弃，输出缩放 |
| BatchNorm | 使用当前批次统计量 | 使用全局运行统计量 |
| 其他层 | 相同 | 相同 |

```python
import torch.nn as nn

# Dropout 在不同模式下的行为
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)

dropout.train()
print(f"训练模式 Dropout: {dropout(x)}")  # 约一半是0

dropout.eval()
print(f"评估模式 Dropout: {dropout(x)}")  # 全部是1（或缩放后的值）
```

## 禁用梯度计算

推理时不需要计算梯度，使用 `torch.no_grad()` 可以：
- 节省内存（不存储中间结果）
- 加速计算

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

x = torch.randn(1, 3, 224, 224)

# 不推荐：会计算并存储梯度
y1 = model(x)
print(f"需要梯度: {y1.requires_grad}")  # True

# 推荐：使用 no_grad
with torch.no_grad():
    y2 = model(x)
print(f"需要梯度: {y2.requires_grad}")  # False
```

### torch.inference_mode()

PyTorch 1.9+ 引入了 `torch.inference_mode()`，比 `no_grad()` 更快：

```python
# 更快的推理模式
with torch.inference_mode():
    y = model(x)
print(f"需要梯度: {y.requires_grad}")  # False
```

## 完整推理流程

### 1. 图像预处理

```python
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

# 加载图像
# image = Image.open("cat.jpg")

# 方法1：手动定义预处理
preprocess_manual = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 方法2：使用权重自带的预处理（推荐）
weights = ResNet50_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# 预处理图像
# input_tensor = preprocess(image)
# input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
```

### 2. 模型推理

```python
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 加载模型和权重
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.eval()

# 获取类别名称
categories = weights.meta["categories"]

# 推理函数
def predict(image_tensor):
    """
    对单张图像进行预测
    
    参数:
        image_tensor: 预处理后的图像张量 (3, 224, 224)
    
    返回:
        top5_probs: 前5个类别的概率
        top5_classes: 前5个类别的名称
    """
    # 添加批次维度
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
    
    # 获取 Top-5 预测
    top5_probs, top5_indices = probs.topk(5, dim=1)
    top5_classes = [categories[idx] for idx in top5_indices[0]]
    
    return top5_probs[0].tolist(), top5_classes

# 使用示例
# probs, classes = predict(input_tensor)
# for prob, cls in zip(probs, classes):
#     print(f"{cls}: {prob:.2%}")
```

### 3. 批量推理

```python
def batch_predict(image_tensors):
    """
    对一批图像进行预测
    
    参数:
        image_tensors: 形状 (batch_size, 3, 224, 224)
    
    返回:
        predictions: 每张图像的预测类别索引
        confidences: 每张图像的预测置信度
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(image_tensors)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
    
    return predictions, confidences

# 示例
batch = torch.randn(8, 3, 224, 224)  # 8张图像
preds, confs = batch_predict(batch)
print(f"预测形状: {preds.shape}")  # [8]
print(f"置信度范围: [{confs.min():.4f}, {confs.max():.4f}]")
```

## GPU 推理

### 将模型和数据移到 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将模型移到 GPU
model = models.resnet50(weights='IMAGENET1K_V1')
model = model.to(device)
model.eval()

# 将数据移到 GPU
x = torch.randn(1, 3, 224, 224).to(device)

# 推理
with torch.no_grad():
    y = model(x)

print(f"输出设备: {y.device}")
```

### GPU 推理性能对比

```python
import time

def benchmark(model, device, batch_size=32, num_iterations=100):
    """基准测试推理速度"""
    model = model.to(device)
    model.eval()
    
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    throughput = batch_size * num_iterations / elapsed
    
    return throughput

# 运行基准测试
model = models.resnet50(weights='IMAGENET1K_V1')

cpu_throughput = benchmark(model, torch.device('cpu'), batch_size=8, num_iterations=10)
print(f"CPU 吞吐量: {cpu_throughput:.1f} images/sec")

if torch.cuda.is_available():
    gpu_throughput = benchmark(model, torch.device('cuda'), batch_size=32, num_iterations=100)
    print(f"GPU 吞吐量: {gpu_throughput:.1f} images/sec")
    print(f"加速比: {gpu_throughput/cpu_throughput:.1f}x")
```

## 混合精度推理

使用 FP16（半精度）可以加速推理并减少内存使用：

```python
# 方法1：使用 autocast
from torch.cuda.amp import autocast

model = models.resnet50(weights='IMAGENET1K_V1').cuda()
model.eval()

x = torch.randn(32, 3, 224, 224).cuda()

with torch.no_grad():
    with autocast():
        y = model(x)

print(f"输出数据类型: {y.dtype}")  # torch.float16

# 方法2：将模型转换为 half 精度
model_half = model.half()
x_half = x.half()

with torch.no_grad():
    y_half = model_half(x_half)
```

## 模型导出与部署

### 导出为 TorchScript

TorchScript 是 PyTorch 模型的序列化格式，可以独立于 Python 运行：

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# 方法1：Tracing（追踪）
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("resnet18_traced.pt")

# 方法2：Scripting（脚本化）
# 适用于有控制流的模型
scripted_model = torch.jit.script(model)
scripted_model.save("resnet18_scripted.pt")

# 加载 TorchScript 模型
loaded_model = torch.jit.load("resnet18_traced.pt")
with torch.no_grad():
    y = loaded_model(example_input)
```

### 导出为 ONNX

ONNX 是跨框架的模型格式：

```python
import torch.onnx

model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("模型已导出为 ONNX 格式")
```

### 使用 ONNX Runtime 推理

```python
# pip install onnxruntime
import onnxruntime as ort
import numpy as np

# 创建推理会话
session = ort.InferenceSession("resnet18.onnx")

# 准备输入
input_name = session.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 推理
output = session.run(None, {input_name: x})
print(f"ONNX Runtime 输出形状: {output[0].shape}")
```

## 推理优化技巧

### 1. 使用 torch.compile()（PyTorch 2.0+）

```python
# PyTorch 2.0+ 的编译优化
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()

# 编译模型
compiled_model = torch.compile(model)

# 首次推理会编译，之后更快
x = torch.randn(32, 3, 224, 224)
with torch.no_grad():
    y = compiled_model(x)
```

### 2. 批量处理

尽可能使用批量推理，充分利用 GPU 并行性：

```python
# 不好：逐个处理
for image in images:
    pred = model(image.unsqueeze(0))

# 好：批量处理
batch = torch.stack(images)
preds = model(batch)
```

### 3. 预分配内存

```python
# 对于固定大小的输入，可以预分配内存
class OptimizedPredictor:
    def __init__(self, model, device, batch_size=32):
        self.model = model.to(device).eval()
        self.device = device
        # 预分配输入缓冲区
        self.input_buffer = torch.empty(batch_size, 3, 224, 224, device=device)
    
    def predict(self, images):
        # 复制数据到预分配的缓冲区
        batch_size = len(images)
        self.input_buffer[:batch_size].copy_(images)
        
        with torch.no_grad():
            return self.model(self.input_buffer[:batch_size])
```

### 4. 禁用 cudnn 自动调优（固定输入大小时）

```python
# 当输入大小固定时，禁用自动调优可能更快
torch.backends.cudnn.benchmark = False
```

## 处理实际图像

### 从文件加载图像

```python
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

def load_and_preprocess(image_path):
    """加载并预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 获取预处理函数
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    
    # 预处理
    input_tensor = preprocess(image)
    
    return input_tensor

# 使用示例
# tensor = load_and_preprocess("path/to/image.jpg")
```

### 从 URL 加载图像

```python
import urllib.request
from PIL import Image
import io

def load_image_from_url(url):
    """从 URL 加载图像"""
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

# 示例
# url = "https://example.com/cat.jpg"
# image = load_image_from_url(url)
```

### 处理视频帧

```python
# pip install opencv-python
import cv2
import torch

def process_video(video_path, model, preprocess, device='cpu'):
    """处理视频的每一帧"""
    model = model.to(device).eval()
    
    cap = cv2.VideoCapture(video_path)
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # 预处理
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        
        results.append(pred)
    
    cap.release()
    return results
```

## 本节小结

| 概念 | 说明 |
|------|------|
| `model.eval()` | 切换到评估模式 |
| `torch.no_grad()` | 禁用梯度计算 |
| `torch.inference_mode()` | 更快的推理模式 |
| 预处理 | Resize → CenterCrop → ToTensor → Normalize |
| GPU 推理 | `model.to(device)`, `x.to(device)` |
| TorchScript | `torch.jit.trace()` 或 `torch.jit.script()` |
| ONNX | `torch.onnx.export()` |

## 练习

1. 加载一个预训练的 ResNet-18，对本地图像进行分类，输出 Top-5 结果
2. 对比 CPU 和 GPU 推理速度，以及不同批大小的影响
3. 将模型导出为 ONNX 格式，使用 ONNX Runtime 进行推理

## 延伸阅读

- [PyTorch 推理模式文档](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)
- [TorchScript 教程](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [ONNX 官网](https://onnx.ai/)

---

[← 上一节：torchvision 模型库](./01_torchvision_models.md) | [下一节：迁移学习 →](./03_transfer_learning.md)

