# 4.2 常用层类型

## 概述

PyTorch 提供了丰富的预定义层，涵盖了深度学习中的各种需求。本节将详细介绍常用的层类型及其物理直觉。

## 线性层 (Linear Layers)

### nn.Linear

全连接层，实现 $y = xW^T + b$：

```python
import torch
import torch.nn as nn

# 输入 10 维，输出 5 维
linear = nn.Linear(in_features=10, out_features=5, bias=True)

x = torch.randn(32, 10)  # batch_size=32
y = linear(x)
print(f"输入: {x.shape}, 输出: {y.shape}")

# 查看参数
print(f"权重形状: {linear.weight.shape}")  # [5, 10]
print(f"偏置形状: {linear.bias.shape}")    # [5]
```

**物理类比**：线性变换，类似于坐标变换或基的变换。

### nn.Bilinear

双线性层，实现 $y = x_1^T A x_2 + b$：

```python
bilinear = nn.Bilinear(in1_features=10, in2_features=20, out_features=5)

x1 = torch.randn(32, 10)
x2 = torch.randn(32, 20)
y = bilinear(x1, x2)
print(f"输出: {y.shape}")  # [32, 5]
```

**物理类比**：用于建模两个输入之间的交互，类似于双线性形式。

## 卷积层 (Convolutional Layers)

### nn.Conv1d

一维卷积，常用于时间序列、音频、光谱数据：

```python
# 输入：(batch, channels, length)
conv1d = nn.Conv1d(
    in_channels=1,      # 输入通道数
    out_channels=16,    # 输出通道数（滤波器数量）
    kernel_size=5,      # 卷积核大小
    stride=1,           # 步长
    padding=2           # 填充
)

x = torch.randn(8, 1, 100)  # 8个样本，1通道，长度100
y = conv1d(x)
print(f"输入: {x.shape}, 输出: {y.shape}")  # [8, 16, 100]
```

**物理应用**：
- 时间序列滤波
- 光谱特征提取
- 一维信号处理

### nn.Conv2d

二维卷积，图像处理的核心：

```python
# 输入：(batch, channels, height, width)
conv2d = nn.Conv2d(
    in_channels=3,      # RGB 图像
    out_channels=64,    # 64 个滤波器
    kernel_size=3,      # 3×3 卷积核
    stride=1,
    padding=1           # 保持尺寸不变
)

x = torch.randn(8, 3, 224, 224)
y = conv2d(x)
print(f"输入: {x.shape}, 输出: {y.shape}")  # [8, 64, 224, 224]
```

**输出尺寸计算**：

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1 \right\rfloor$$

```python
# 常用配置
# kernel=3, padding=1, stride=1 → 尺寸不变
# kernel=3, padding=1, stride=2 → 尺寸减半
# kernel=1, padding=0, stride=1 → 1×1 卷积，只改变通道数
```

### nn.Conv3d

三维卷积，用于视频、体积数据：

```python
# 输入：(batch, channels, depth, height, width)
conv3d = nn.Conv3d(
    in_channels=1,
    out_channels=32,
    kernel_size=(3, 3, 3),
    padding=1
)

x = torch.randn(2, 1, 16, 64, 64)  # 医学影像体数据
y = conv3d(x)
print(f"输入: {x.shape}, 输出: {y.shape}")
```

**物理应用**：
- 医学影像（CT、MRI）
- 流体动力学模拟数据
- 时空场分析

### 转置卷积 (反卷积)

用于上采样：

```python
# 转置卷积（上采样）
conv_transpose = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1
)

x = torch.randn(8, 64, 16, 16)
y = conv_transpose(x)
print(f"输入: {x.shape}, 输出: {y.shape}")  # [8, 32, 32, 32] 尺寸翻倍
```

## 池化层 (Pooling Layers)

### 最大池化

```python
# 2D 最大池化
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(8, 64, 32, 32)
y = maxpool(x)
print(f"输入: {x.shape}, 输出: {y.shape}")  # [8, 64, 16, 16]

# 同时返回索引（用于反池化）
maxpool_with_indices = nn.MaxPool2d(2, 2, return_indices=True)
y, indices = maxpool_with_indices(x)
```

### 平均池化

```python
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
y = avgpool(x)
```

### 自适应池化

输出固定大小，无论输入大小：

```python
# 输出固定为 7×7
adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(7, 7))
adaptive_max = nn.AdaptiveMaxPool2d(output_size=(7, 7))

x = torch.randn(8, 64, 224, 224)
y = adaptive_avg(x)
print(f"输入: {x.shape}, 输出: {y.shape}")  # [8, 64, 7, 7]

# 全局平均池化
global_avg = nn.AdaptiveAvgPool2d(1)
y = global_avg(x)
print(f"全局池化: {y.shape}")  # [8, 64, 1, 1]
```

**物理直觉**：池化类似于粗粒化（coarse-graining），保留主要特征，忽略细节。

## 归一化层 (Normalization Layers)

### 批归一化 (Batch Normalization)

```python
# 用于全连接层
bn1d = nn.BatchNorm1d(num_features=64)

# 用于卷积层
bn2d = nn.BatchNorm2d(num_features=64)

x = torch.randn(32, 64, 16, 16)
y = bn2d(x)

print(f"归一化后均值: {y.mean():.4f}")  # 接近 0
print(f"归一化后标准差: {y.std():.4f}")  # 接近 1
```

**工作原理**：
- 训练时：使用当前批次的均值和方差
- 推理时：使用训练期间累积的运行统计量

### 层归一化 (Layer Normalization)

对每个样本独立归一化，不依赖批次：

```python
# 归一化最后 N 个维度
ln = nn.LayerNorm(normalized_shape=[64, 16, 16])

x = torch.randn(32, 64, 16, 16)
y = ln(x)

# 常用于 Transformer
ln_transformer = nn.LayerNorm(512)  # 归一化特征维度
```

### 实例归一化 (Instance Normalization)

对每个样本的每个通道独立归一化：

```python
instance_norm = nn.InstanceNorm2d(num_features=64)
y = instance_norm(x)
```

**用途**：风格迁移等任务。

### 组归一化 (Group Normalization)

将通道分组后归一化：

```python
# 将 64 个通道分成 8 组
group_norm = nn.GroupNorm(num_groups=8, num_channels=64)
y = group_norm(x)
```

**优点**：不依赖批大小，适合小批量训练。

### 归一化层对比

| 类型 | 归一化维度 | 依赖批大小 | 典型用途 |
|------|-----------|-----------|---------|
| BatchNorm | 批次内同一通道 | 是 | CNN |
| LayerNorm | 单样本所有特征 | 否 | Transformer |
| InstanceNorm | 单样本单通道 | 否 | 风格迁移 |
| GroupNorm | 单样本通道组 | 否 | 小批量 |

## 循环层 (Recurrent Layers)

### nn.RNN

基础循环神经网络：

```python
# 输入：(seq_len, batch, input_size) 或 (batch, seq_len, input_size)
rnn = nn.RNN(
    input_size=10,      # 输入特征维度
    hidden_size=20,     # 隐藏状态维度
    num_layers=2,       # 层数
    batch_first=True,   # 输入格式 (batch, seq, feature)
    bidirectional=False # 单向/双向
)

x = torch.randn(32, 50, 10)  # 32个样本，序列长度50，特征维度10
output, h_n = rnn(x)

print(f"输出: {output.shape}")  # [32, 50, 20]
print(f"最终隐状态: {h_n.shape}")  # [2, 32, 20] (num_layers, batch, hidden)
```

### nn.LSTM

长短期记忆网络，解决长程依赖问题：

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,        # 层间 dropout
    bidirectional=True  # 双向
)

x = torch.randn(32, 50, 10)
output, (h_n, c_n) = lstm(x)

print(f"输出: {output.shape}")  # [32, 50, 40] (双向所以是 20*2)
print(f"隐状态: {h_n.shape}")   # [4, 32, 20] (2层*2方向)
print(f"细胞状态: {c_n.shape}") # [4, 32, 20]
```

### nn.GRU

门控循环单元，LSTM 的简化版：

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(32, 50, 10)
output, h_n = gru(x)
```

**物理应用**：
- 时间序列预测（温度、压力等）
- 动力学系统建模
- 序列到序列映射

## 注意力层 (Attention Layers)

### nn.MultiheadAttention

多头自注意力机制：

```python
# embed_dim 必须能被 num_heads 整除
mha = nn.MultiheadAttention(
    embed_dim=512,      # 嵌入维度
    num_heads=8,        # 注意力头数
    dropout=0.1,
    batch_first=True
)

# 自注意力：Q, K, V 相同
x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
attn_output, attn_weights = mha(x, x, x)

print(f"注意力输出: {attn_output.shape}")  # [32, 100, 512]
print(f"注意力权重: {attn_weights.shape}") # [32, 100, 100]

# 交叉注意力：Q 来自一个序列，K, V 来自另一个
query = torch.randn(32, 50, 512)
key_value = torch.randn(32, 100, 512)
attn_output, _ = mha(query, key_value, key_value)
print(f"交叉注意力输出: {attn_output.shape}")  # [32, 50, 512]
```

### 带掩码的注意力

```python
# 创建因果掩码（防止看到未来）
seq_len = 100
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# 应用掩码
attn_output, _ = mha(x, x, x, attn_mask=causal_mask)
```

## Dropout 层

### 标准 Dropout

```python
dropout = nn.Dropout(p=0.5)

x = torch.randn(10, 10)
print(f"训练模式输出:\n{dropout(x)}")

dropout.eval()
print(f"评估模式输出:\n{dropout(x)}")  # 不 dropout
```

### 其他 Dropout 变体

```python
# 1D: 随机丢弃整个通道
dropout1d = nn.Dropout1d(p=0.5)

# 2D: 随机丢弃整个特征图
dropout2d = nn.Dropout2d(p=0.5)

# 3D: 随机丢弃整个 3D 特征
dropout3d = nn.Dropout3d(p=0.5)

# Alpha Dropout: 用于 SELU 激活
alpha_dropout = nn.AlphaDropout(p=0.5)
```

## 激活函数层

```python
# 常用激活函数
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
prelu = nn.PReLU()  # 可学习的斜率
elu = nn.ELU()
selu = nn.SELU()
gelu = nn.GELU()
silu = nn.SiLU()  # Swish
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)
```

## 嵌入层 (Embedding Layer)

将离散索引映射为连续向量：

```python
# 创建嵌入层：10000个词，每个词映射为256维向量
embedding = nn.Embedding(
    num_embeddings=10000,  # 词表大小
    embedding_dim=256,     # 嵌入维度
    padding_idx=0          # 填充索引（输出全零）
)

# 输入是索引
indices = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # (batch, seq_len)
embedded = embedding(indices)
print(f"嵌入输出: {embedded.shape}")  # [2, 4, 256]
```

**物理应用**：
- 原子类型嵌入（分子动力学）
- 粒子类型嵌入

## 展平与重塑

```python
# 展平
flatten = nn.Flatten()
x = torch.randn(32, 64, 7, 7)
y = flatten(x)
print(f"展平: {x.shape} -> {y.shape}")  # [32, 3136]

# 指定起始维度
flatten_partial = nn.Flatten(start_dim=1, end_dim=-1)

# Unflatten
unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
y_restored = unflatten(y)
print(f"恢复: {y.shape} -> {y_restored.shape}")  # [32, 64, 7, 7]
```

## 物理应用示例

### 示例1：一维卷积处理光谱数据

```python
class SpectralCNN(nn.Module):
    """处理一维光谱数据的 CNN"""
    
    def __init__(self, spectrum_length, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, spectrum_length)
        x = x.unsqueeze(1)  # (batch, 1, spectrum_length)
        x = self.features(x)
        x = x.squeeze(-1)   # (batch, 128)
        return self.classifier(x)

model = SpectralCNN(spectrum_length=1000, num_classes=5)
x = torch.randn(8, 1000)
y = model(x)
print(f"输出: {y.shape}")  # [8, 5]
```

### 示例2：LSTM 处理时间序列

```python
class TimeSeriesLSTM(nn.Module):
    """处理物理时间序列的 LSTM"""
    
    def __init__(self, input_features, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# 预测物理系统的未来状态
model = TimeSeriesLSTM(
    input_features=6,   # 位置和速度 (x, y, z, vx, vy, vz)
    hidden_size=64,
    num_layers=2,
    output_size=3       # 预测下一个位置
)

x = torch.randn(16, 100, 6)  # 100 个时间步
y = model(x)
print(f"输出: {y.shape}")  # [16, 3]
```

## 本节小结

| 层类型 | 主要用途 | 输入形状 |
|--------|---------|---------|
| Linear | 全连接 | (batch, features) |
| Conv1d/2d/3d | 局部特征提取 | (batch, channels, ...) |
| MaxPool/AvgPool | 下采样 | 同卷积 |
| BatchNorm/LayerNorm | 归一化 | 取决于类型 |
| RNN/LSTM/GRU | 序列建模 | (batch, seq, features) |
| MultiheadAttention | 注意力机制 | (batch, seq, embed_dim) |
| Embedding | 离散→连续 | (batch, seq) 整数 |
| Dropout | 正则化 | 任意 |

## 练习

1. 实现一个 2D CNN 用于处理 $64 \times 64$ 的物理场图像
2. 使用 LSTM 预测阻尼振子的轨迹
3. 实现一个使用自注意力的序列分类器

## 延伸阅读

- [PyTorch nn 文档](https://pytorch.org/docs/stable/nn.html)
- LeCun et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"

---

[← 上一节：nn.Module 详解](./01_nn_module.md) | [下一节：模型设计模式 →](./03_model_design_patterns.md)

