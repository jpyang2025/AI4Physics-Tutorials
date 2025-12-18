# 5.3 训练循环

## 📖 概述

训练循环是深度学习的核心流程，将数据加载、前向传播、损失计算、反向传播和参数更新整合在一起。本节介绍如何编写高效、可靠的训练循环。

## 🎯 学习目标

- 掌握标准训练循环的结构
- 理解训练过程中的关键步骤
- 实现训练日志和进度监控
- 处理常见的训练问题

---

## 5.3.1 基本训练循环

### 最简训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def basic_training_loop(model, train_loader, criterion, optimizer, num_epochs):
    """
    最基本的训练循环
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
    """
    model.train()  # 设置为训练模式
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 1. 清空梯度
            optimizer.zero_grad()
            
            # 2. 前向传播
            outputs = model(inputs)
            
            # 3. 计算损失
            loss = criterion(outputs, targets)
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新参数
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
```

### 训练循环详解

```
训练一个 Epoch 的流程：

┌────────────────────────────────────────────────────────────┐
│ for batch in dataloader:                                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ① optimizer.zero_grad()                                  │
│     └─ 清空上一步的梯度，防止累积                          │
│                                                            │
│  ② outputs = model(inputs)                                │
│     └─ 前向传播：计算预测值                                │
│                                                            │
│  ③ loss = criterion(outputs, targets)                     │
│     └─ 计算损失：比较预测与真实值                          │
│                                                            │
│  ④ loss.backward()                                        │
│     └─ 反向传播：计算所有参数的梯度                        │
│                                                            │
│  ⑤ optimizer.step()                                       │
│     └─ 参数更新：根据梯度更新权重                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 5.3.2 完整训练流程

### 包含验证的训练循环

```python
def train_with_validation(model, train_loader, val_loader, 
                         criterion, optimizer, num_epochs, device='cpu'):
    """
    包含验证的完整训练流程
    """
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    return history
```

---

## 5.3.3 GPU 训练

### 设备管理

```python
# 检测可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 如果有多个 GPU
if torch.cuda.is_available():
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
```

### GPU 训练循环

```python
def gpu_training_loop(model, train_loader, criterion, optimizer, 
                      num_epochs, device):
    """GPU 训练循环"""
    
    # 模型移动到 GPU
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            # 数据移动到 GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
```

### 混合精度训练

使用 AMP（Automatic Mixed Precision）加速训练并减少显存占用。

```python
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training(model, train_loader, criterion, optimizer, 
                             num_epochs, device):
    """混合精度训练"""
    
    model = model.to(device)
    scaler = GradScaler()  # 梯度缩放器
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
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
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
```

---

## 5.3.4 进度监控

### 使用 tqdm 进度条

```python
from tqdm import tqdm

def training_with_progress(model, train_loader, criterion, optimizer, 
                          num_epochs, device):
    """带进度条的训练"""
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条显示
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
```

### TensorBoard 日志

```python
from torch.utils.tensorboard import SummaryWriter

def training_with_tensorboard(model, train_loader, val_loader,
                              criterion, optimizer, num_epochs, device):
    """使用 TensorBoard 记录训练过程"""
    
    writer = SummaryWriter('runs/experiment_1')
    model = model.to(device)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 记录每步损失
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
        
        # 记录每个 epoch 的平均损失
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        
        # 记录学习率
        writer.add_scalar('Learning_rate', 
                         optimizer.param_groups[0]['lr'], epoch)
        
        # 记录模型参数直方图
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}")
    
    writer.close()

# 启动 TensorBoard: tensorboard --logdir=runs
```

---

## 5.3.5 训练技巧

### 早停（Early Stopping）

```python
class EarlyStopping:
    """早停机制，防止过拟合"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: 等待改善的轮数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


# 使用早停
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    # 训练...
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print(f"早停于 Epoch {epoch+1}")
        break
```

### 模型检查点

```python
class ModelCheckpoint:
    """模型检查点，保存最佳模型"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, score, model, optimizer=None, epoch=None):
        if self._is_improvement(score):
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, score)
            return True
        return False
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score
        return score > self.best_score
    
    def _save_checkpoint(self, model, optimizer, epoch, score):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, self.filepath)
        print(f"✓ 保存检查点: {self.monitor}={score:.4f}")


# 使用检查点
checkpoint = ModelCheckpoint('best_model.pth', monitor='val_loss')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    checkpoint(val_loss, model, optimizer, epoch)
```

### 梯度监控

```python
def monitor_gradients(model, log_interval=100):
    """梯度监控装饰器"""
    
    gradient_norms = []
    
    def hook(grad):
        gradient_norms.append(grad.norm().item())
    
    # 注册钩子
    hooks = []
    for param in model.parameters():
        if param.requires_grad:
            hooks.append(param.register_hook(hook))
    
    return hooks, gradient_norms


# 训练中监控梯度
def train_with_gradient_monitoring(model, train_loader, criterion, 
                                   optimizer, device):
    """带梯度监控的训练"""
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 100:
            print(f"⚠️ 警告: 梯度范数过大 ({total_norm:.2f})")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
```

---

## 5.3.6 完整训练器类

```python
class Trainer:
    """通用训练器类"""
    
    def __init__(self, model, criterion, optimizer, device='cpu',
                 scheduler=None, early_stopping=None, checkpoint=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            
            # 计算准确率（分类任务）
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, num_epochs):
        """完整训练流程"""
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, 
                             torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 打印进度
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                  f"LR: {lr:.2e}")
            
            # 检查点
            if self.checkpoint is not None:
                self.checkpoint(val_loss, self.model, self.optimizer, epoch)
            
            # 早停
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        return self.history
    
    def plot_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')
        
        # 准确率曲线
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].set_title('Accuracy Curve')
        
        plt.tight_layout()
        plt.show()


# 使用示例
trainer = Trainer(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    device=device,
    early_stopping=EarlyStopping(patience=10),
    checkpoint=ModelCheckpoint('best_model.pth')
)

history = trainer.fit(train_loader, val_loader, num_epochs=100)
trainer.plot_history()
```

---

## 5.3.7 物理问题训练示例

### 训练神经网络求解微分方程

```python
def train_physics_network():
    """
    训练神经网络求解常微分方程
    
    问题：y'' + y = 0
    边界条件：y(0) = 0, y(π) = 0
    解析解：y(x) = A*sin(x)
    """
    
    class PhysicsNet(nn.Module):
        """满足边界条件的网络"""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            # 自动满足边界条件 y(0)=0, y(π)=0
            # y = x(π-x) * NN(x)
            return x * (torch.pi - x) * self.net(x)
    
    model = PhysicsNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 内部配点
    x_interior = torch.linspace(0.01, torch.pi - 0.01, 100, 
                                 requires_grad=True).reshape(-1, 1)
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        # 计算 y, y', y''
        y = model(x_interior)
        
        y_x = torch.autograd.grad(
            y, x_interior, 
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        y_xx = torch.autograd.grad(
            y_x, x_interior,
            grad_outputs=torch.ones_like(y_x),
            create_graph=True
        )[0]
        
        # 残差：y'' + y = 0
        residual = y_xx + y
        loss = (residual ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Residual Loss: {loss.item():.6f}")
    
    return model


# 训练
model = train_physics_network()

# 验证
x_test = torch.linspace(0, torch.pi, 100).reshape(-1, 1)
with torch.no_grad():
    y_pred = model(x_test)

# 归一化后比较
y_pred_normalized = y_pred / y_pred.max()
y_exact = torch.sin(x_test)
print(f"最大误差: {(y_pred_normalized - y_exact).abs().max().item():.6f}")
```

---

## 🔬 物理视角总结

### 训练作为动力学演化

训练过程可以看作参数空间中的动力学演化：

$$\theta(t+\Delta t) = \theta(t) - \eta \nabla L(\theta(t))$$

这是离散化的梯度流方程。

### Epoch 的物理意义

- **一个 Batch**：一次力的测量（有噪声）
- **一个 Epoch**：遍历整个相空间
- **多个 Epoch**：系统向平衡态弛豫

### 收敛判据

类似于物理系统达到平衡的判据：

| 物理系统 | 神经网络训练 |
|---------|-------------|
| 能量不再下降 | 损失不再下降 |
| 涨落变小 | 梯度范数变小 |
| 温度降低 | 学习率衰减 |

---

## 📝 练习

1. 实现一个完整的训练循环，包含早停和检查点保存
2. 使用 TensorBoard 记录训练过程
3. 实现混合精度训练并比较速度

---

## ⏭️ 下一节

下一节我们将学习 [验证与测试](./04_validation_testing.md)，了解如何评估模型性能。

