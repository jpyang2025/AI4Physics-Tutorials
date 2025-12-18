#!/usr/bin/env python3
"""
学习率调度器演示

本示例演示各种学习率调度策略：
1. StepLR 和 MultiStepLR
2. ExponentialLR
3. CosineAnnealingLR
4. ReduceLROnPlateau
5. OneCycleLR
6. 学习率预热
7. 自定义调度器

作者：PyTorch 教程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CyclicLR, LambdaLR,
    CosineAnnealingWarmRestarts
)
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. 学习率调度器可视化
# =============================================================================

def visualize_all_schedulers():
    """可视化所有常用学习率调度器"""
    print("\n" + "=" * 60)
    print("可视化学习率调度器")
    print("=" * 60)
    
    epochs = 100
    initial_lr = 0.1
    steps_per_epoch = 100  # 用于 OneCycleLR
    
    # 定义调度器
    def create_schedulers():
        schedulers = {}
        
        # StepLR
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['StepLR (step=30, γ=0.1)'] = {
            'optimizer': opt,
            'scheduler': StepLR(opt, step_size=30, gamma=0.1),
            'per_epoch': True
        }
        
        # MultiStepLR
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['MultiStepLR'] = {
            'optimizer': opt,
            'scheduler': MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.1),
            'per_epoch': True
        }
        
        # ExponentialLR
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['ExponentialLR (γ=0.95)'] = {
            'optimizer': opt,
            'scheduler': ExponentialLR(opt, gamma=0.95),
            'per_epoch': True
        }
        
        # CosineAnnealingLR
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['CosineAnnealingLR'] = {
            'optimizer': opt,
            'scheduler': CosineAnnealingLR(opt, T_max=epochs),
            'per_epoch': True
        }
        
        # CosineAnnealingWarmRestarts
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['CosineAnnealingWarmRestarts'] = {
            'optimizer': opt,
            'scheduler': CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2),
            'per_epoch': True
        }
        
        # CyclicLR
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        schedulers['CyclicLR (triangular)'] = {
            'optimizer': opt,
            'scheduler': CyclicLR(opt, base_lr=0.001, max_lr=0.1, 
                                   step_size_up=1000, mode='triangular'),
            'per_epoch': False,
            'total_steps': epochs * steps_per_epoch
        }
        
        return schedulers
    
    schedulers = create_schedulers()
    
    # 收集学习率
    lr_curves = {}
    
    for name, config in schedulers.items():
        opt = config['optimizer']
        sched = config['scheduler']
        lrs = []
        
        if config['per_epoch']:
            for epoch in range(epochs):
                lrs.append(opt.param_groups[0]['lr'])
                sched.step()
        else:
            for step in range(config['total_steps']):
                if step % steps_per_epoch == 0:
                    lrs.append(opt.param_groups[0]['lr'])
                sched.step()
        
        lr_curves[name] = lrs
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for ax, (name, lrs) in zip(axes.flat, lr_curves.items()):
        ax.plot(lrs, linewidth=2, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('lr_schedulers_visualization.png', dpi=150)
    plt.show()


def visualize_warmup():
    """可视化学习率预热"""
    print("\n" + "=" * 60)
    print("可视化学习率预热")
    print("=" * 60)
    
    epochs = 100
    warmup_epochs = 10
    initial_lr = 0.1
    
    # 线性预热
    def linear_warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    # 线性预热 + 余弦衰减
    def warmup_cosine_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    # 收集学习率
    warmup_types = {
        '无预热': lambda e: 1.0,
        '线性预热': linear_warmup_lambda,
        '线性预热 + 余弦衰减': warmup_cosine_lambda,
    }
    
    lr_curves = {}
    
    for name, lr_lambda in warmup_types.items():
        opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
        scheduler = LambdaLR(opt, lr_lambda)
        
        lrs = []
        for epoch in range(epochs):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        lr_curves[name] = lrs
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    for name, lrs in lr_curves.items():
        plt.plot(lrs, label=name, linewidth=2)
    
    plt.axvline(x=warmup_epochs, color='gray', linestyle='--', label='预热结束')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('学习率预热策略')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_warmup_visualization.png', dpi=150)
    plt.show()


# =============================================================================
# 2. 训练实验
# =============================================================================

def generate_data(n_samples=1000, n_features=20):
    """生成回归数据"""
    X = torch.randn(n_samples, n_features)
    w = torch.randn(n_features, 1)
    y = X @ w + 0.1 * torch.randn(n_samples, 1)
    return X, y


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def train_with_scheduler(model, train_loader, val_loader, optimizer, 
                         scheduler, num_epochs, scheduler_per_batch=False):
    """使用指定调度器训练模型"""
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(num_epochs):
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
            
            if scheduler_per_batch and scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        if not scheduler_per_batch and scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
    return history


def compare_schedulers():
    """比较不同学习率调度器的训练效果"""
    print("\n" + "=" * 60)
    print("比较不同学习率调度器")
    print("=" * 60)
    
    # 数据
    X, y = generate_data(1000, 20)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    initial_lr = 0.1
    
    # 不同调度器配置
    def create_configs():
        configs = []
        
        # 无调度
        model = SimpleNet(20).to(device)
        opt = optim.SGD(model.parameters(), lr=initial_lr)
        configs.append({
            'name': '恒定学习率',
            'model': model,
            'optimizer': opt,
            'scheduler': None,
            'per_batch': False
        })
        
        # StepLR
        model = SimpleNet(20).to(device)
        opt = optim.SGD(model.parameters(), lr=initial_lr)
        configs.append({
            'name': 'StepLR',
            'model': model,
            'optimizer': opt,
            'scheduler': StepLR(opt, step_size=15, gamma=0.1),
            'per_batch': False
        })
        
        # CosineAnnealingLR
        model = SimpleNet(20).to(device)
        opt = optim.SGD(model.parameters(), lr=initial_lr)
        configs.append({
            'name': 'CosineAnnealingLR',
            'model': model,
            'optimizer': opt,
            'scheduler': CosineAnnealingLR(opt, T_max=num_epochs),
            'per_batch': False
        })
        
        # ReduceLROnPlateau
        model = SimpleNet(20).to(device)
        opt = optim.SGD(model.parameters(), lr=initial_lr)
        configs.append({
            'name': 'ReduceLROnPlateau',
            'model': model,
            'optimizer': opt,
            'scheduler': ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5),
            'per_batch': False
        })
        
        return configs
    
    configs = create_configs()
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        
        history = train_with_scheduler(
            config['model'], train_loader, val_loader,
            config['optimizer'], config['scheduler'],
            num_epochs, config['per_batch']
        )
        
        results[config['name']] = history
        print(f"  最终验证损失: {history['val_loss'][-1]:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 训练损失
    for name, history in results.items():
        axes[0].plot(history['train_loss'], label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('训练损失')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # 验证损失
    for name, history in results.items():
        axes[1].plot(history['val_loss'], label=name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('验证损失')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    # 学习率
    for name, history in results.items():
        axes[2].plot(history['lr'], label=name)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('学习率变化')
    axes[2].legend()
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('scheduler_comparison.png', dpi=150)
    plt.show()


# =============================================================================
# 3. 学习率查找器
# =============================================================================

def learning_rate_finder():
    """学习率范围测试"""
    print("\n" + "=" * 60)
    print("学习率查找器")
    print("=" * 60)
    
    # 数据
    X, y = generate_data(1000, 20)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleNet(20).to(device)
    criterion = nn.MSELoss()
    
    # 参数
    init_lr = 1e-7
    final_lr = 10
    num_steps = 100
    
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    
    # 保存初始状态
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if loss.item() > 4 * best_loss:
            print("损失发散，停止搜索")
            break
        
        loss.backward()
        optimizer.step()
        
        for pg in optimizer.param_groups:
            pg['lr'] *= lr_mult
    
    # 恢复初始状态
    model.load_state_dict(initial_state)
    
    # 找到最佳学习率
    gradients = np.gradient(losses)
    best_idx = np.argmin(gradients)
    suggested_lr = lrs[best_idx]
    
    print(f"建议学习率: {suggested_lr:.2e}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].semilogx(lrs, losses)
    axes[0].axvline(x=suggested_lr, color='r', linestyle='--', 
                    label=f'建议: {suggested_lr:.2e}')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Learning Rate Finder')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(lrs[:len(gradients)], gradients)
    axes[1].axvline(x=suggested_lr, color='r', linestyle='--')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Loss Gradient')
    axes[1].set_title('损失梯度')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_finder.png', dpi=150)
    plt.show()
    
    return suggested_lr


# =============================================================================
# 4. 自定义调度器
# =============================================================================

def demo_custom_scheduler():
    """演示自定义调度器"""
    print("\n" + "=" * 60)
    print("自定义调度器演示")
    print("=" * 60)
    
    epochs = 100
    initial_lr = 0.1
    
    # 自定义调度函数
    def simulated_annealing(epoch, T0=1.0, T_min=0.01, alpha=0.95):
        """模拟退火调度"""
        return max(T_min, T0 * (alpha ** epoch))
    
    def warmup_then_decay(epoch, warmup=10, decay_start=30):
        """预热 + 阶梯衰减"""
        if epoch < warmup:
            return (epoch + 1) / warmup
        elif epoch < decay_start:
            return 1.0
        else:
            return 0.1 ** ((epoch - decay_start) / 30)
    
    def cosine_with_restarts(epoch, T_0=20, T_mult=2):
        """带重启的余弦"""
        T_cur = epoch
        T_i = T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= T_mult
        return 0.5 * (1 + np.cos(np.pi * T_cur / T_i))
    
    # 收集学习率
    custom_schedulers = {
        '模拟退火': simulated_annealing,
        '预热 + 阶梯衰减': warmup_then_decay,
        '余弦重启': cosine_with_restarts,
    }
    
    lr_curves = {}
    
    for name, lr_func in custom_schedulers.items():
        lrs = [initial_lr * lr_func(e) for e in range(epochs)]
        lr_curves[name] = lrs
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    for name, lrs in lr_curves.items():
        plt.plot(lrs, label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('自定义学习率调度器')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('custom_schedulers.png', dpi=150)
    plt.show()


# =============================================================================
# 5. OneCycleLR 演示
# =============================================================================

def demo_one_cycle_lr():
    """演示 OneCycleLR"""
    print("\n" + "=" * 60)
    print("OneCycleLR 演示")
    print("=" * 60)
    
    # 数据
    X, y = generate_data(1000, 20)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 30
    steps_per_epoch = len(train_loader)
    max_lr = 0.1
    
    model = SimpleNet(20).to(device)
    optimizer = optim.SGD(model.parameters(), lr=max_lr/25)  # 初始 lr
    criterion = nn.MSELoss()
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000
    )
    
    # 训练并记录学习率
    lrs = []
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            lrs.append(scheduler.get_last_lr()[0])
            losses.append(loss.item())
            
            scheduler.step()
        
        epoch_loss = np.mean(losses[-steps_per_epoch:])
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(lrs)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('OneCycleLR 学习率曲线')
    axes[0].grid(True, alpha=0.3)
    
    # 平滑损失
    window = 50
    smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_losses)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss (smoothed)')
    axes[1].set_title('训练损失')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('one_cycle_lr.png', dpi=150)
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有演示"""
    print("学习率调度器演示")
    print("=" * 60)
    
    visualize_all_schedulers()
    visualize_warmup()
    compare_schedulers()
    learning_rate_finder()
    demo_custom_scheduler()
    demo_one_cycle_lr()
    
    print("\n所有演示完成！")


def quick_demo():
    """快速演示"""
    print("学习率调度器快速演示")
    print("=" * 60)
    
    # 可视化几个常用调度器
    epochs = 50
    initial_lr = 0.1
    
    schedulers = {}
    
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['StepLR'] = StepLR(opt, step_size=15, gamma=0.1)
    
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=initial_lr)
    schedulers['CosineAnnealing'] = CosineAnnealingLR(opt, T_max=epochs)
    
    plt.figure(figsize=(10, 4))
    
    for name, scheduler in schedulers.items():
        opt = scheduler.optimizer
        lrs = []
        for _ in range(epochs):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        plt.plot(lrs, label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('常用学习率调度器')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('lr_schedulers_quick.png', dpi=150)
    plt.show()
    
    print("\n快速演示完成！")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        main()

