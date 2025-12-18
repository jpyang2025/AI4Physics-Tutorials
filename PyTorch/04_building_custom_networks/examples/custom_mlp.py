#!/usr/bin/env python3
"""
自定义多层感知机 (MLP) 示例

本脚本演示如何使用 nn.Module 构建灵活的多层感知机，
包括动态层数、多种激活函数和正则化技术。

运行方式：
    python custom_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable
import math


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 基础 MLP
# ============================================================

class BasicMLP(nn.Module):
    """最基础的多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_basic_mlp():
    """演示基础 MLP"""
    print_section("1. 基础 MLP")
    
    model = BasicMLP(input_dim=10, hidden_dim=64, output_dim=5)
    
    print(f"模型结构:\n{model}")
    print(f"\n参数统计:")
    for name, param in model.named_parameters():
        print(f"  {name:15} {str(list(param.shape)):15} 参数量: {param.numel():,}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 测试前向传播
    x = torch.randn(8, 10)
    y = model(x)
    print(f"\n前向传播: {x.shape} -> {y.shape}")


# ============================================================
# 2. 可配置的 MLP
# ============================================================

class ConfigurableMLP(nn.Module):
    """
    可配置的多层感知机
    
    支持:
    - 可变数量的隐藏层
    - 多种激活函数
    - Dropout 正则化
    - 批归一化
    """
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 获取激活函数
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"不支持的激活函数: {activation}")
        activation_fn = self.ACTIVATIONS[activation]
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(activation_fn())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation is not None:
            layers.append(self.ACTIVATIONS[output_activation]())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def count_parameters(self) -> dict:
        """统计参数"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def demo_configurable_mlp():
    """演示可配置 MLP"""
    print_section("2. 可配置 MLP")
    
    # 创建不同配置的模型
    configs = [
        {'hidden_dims': [64, 64], 'activation': 'relu', 'dropout': 0.0, 'batch_norm': False},
        {'hidden_dims': [128, 64, 32], 'activation': 'gelu', 'dropout': 0.2, 'batch_norm': True},
        {'hidden_dims': [256, 256, 256, 256], 'activation': 'leaky_relu', 'dropout': 0.1, 'batch_norm': True},
    ]
    
    for i, config in enumerate(configs):
        model = ConfigurableMLP(
            input_dim=10,
            output_dim=5,
            **config
        )
        params = model.count_parameters()
        print(f"配置 {i+1}: 隐藏层={config['hidden_dims']}, "
              f"激活={config['activation']}, 参数量={params['total']:,}")
    
    # 详细展示一个模型
    print("\n详细模型结构 (配置2):")
    model = ConfigurableMLP(
        input_dim=10,
        hidden_dims=[128, 64, 32],
        output_dim=5,
        activation='gelu',
        dropout=0.2,
        batch_norm=True
    )
    print(model.network)


# ============================================================
# 3. 残差 MLP
# ============================================================

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class ResidualMLP(nn.Module):
    """带残差连接的 MLP"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def demo_residual_mlp():
    """演示残差 MLP"""
    print_section("3. 残差 MLP")
    
    model = ResidualMLP(
        input_dim=10,
        hidden_dim=64,
        output_dim=5,
        num_blocks=4,
        dropout=0.1
    )
    
    print(f"模型结构:\n{model}")
    
    # 测试
    x = torch.randn(8, 10)
    y = model(x)
    print(f"\n前向传播: {x.shape} -> {y.shape}")
    
    # 梯度流验证
    y.sum().backward()
    print("\n梯度流验证:")
    print(f"  输入投影梯度范数: {model.input_proj.weight.grad.norm():.4f}")
    print(f"  第一个残差块梯度范数: {model.blocks[0].block[0].weight.grad.norm():.4f}")
    print(f"  最后一个残差块梯度范数: {model.blocks[-1].block[0].weight.grad.norm():.4f}")


# ============================================================
# 4. 多头 MLP (多任务)
# ============================================================

class MultiHeadMLP(nn.Module):
    """
    多头 MLP：共享主干，不同任务头
    
    适用于多任务学习
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        head_configs: dict  # {'task_name': output_dim}
    ):
        super().__init__()
        
        # 共享主干
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.backbone_dim = prev_dim
        
        # 任务头
        self.heads = nn.ModuleDict({
            name: nn.Linear(prev_dim, output_dim)
            for name, output_dim in head_configs.items()
        })
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None):
        """
        前向传播
        
        如果指定 task，返回该任务的输出
        否则返回所有任务的输出字典
        """
        features = self.backbone(x)
        
        if task is not None:
            return self.heads[task](features)
        else:
            return {name: head(features) for name, head in self.heads.items()}
    
    def get_task_names(self) -> List[str]:
        return list(self.heads.keys())


def demo_multihead_mlp():
    """演示多头 MLP"""
    print_section("4. 多头 MLP (多任务)")
    
    model = MultiHeadMLP(
        input_dim=100,
        hidden_dims=[256, 128],
        head_configs={
            'classification': 10,
            'regression': 1,
            'embedding': 64
        }
    )
    
    print(f"任务: {model.get_task_names()}")
    print(f"主干输出维度: {model.backbone_dim}")
    
    x = torch.randn(8, 100)
    
    # 获取所有输出
    outputs = model(x)
    print("\n所有任务输出:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # 获取单个任务输出
    cls_output = model(x, task='classification')
    print(f"\n单任务输出 (classification): {cls_output.shape}")


# ============================================================
# 5. 物理约束 MLP
# ============================================================

class PhysicsConstrainedMLP(nn.Module):
    """
    物理约束 MLP
    
    示例：预测满足某些物理约束的输出
    - 输出非负（如能量、概率密度）
    - 输出归一化（如概率分布）
    - 输出对称（如偶函数）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        constraint: str = 'none'
    ):
        super().__init__()
        
        self.constraint = constraint
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()  # 使用 Tanh 保持有界
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础输出
        output = self.network(x)
        
        # 应用物理约束
        if self.constraint == 'positive':
            # 确保输出非负
            output = F.softplus(output)
        
        elif self.constraint == 'normalized':
            # 确保输出和为 1
            output = F.softmax(output, dim=-1)
        
        elif self.constraint == 'unit_norm':
            # 确保输出范数为 1
            output = F.normalize(output, p=2, dim=-1)
        
        elif self.constraint == 'bounded':
            # 确保输出在 [0, 1] 范围内
            output = torch.sigmoid(output)
        
        elif self.constraint == 'symmetric':
            # 偶函数约束：f(-x) = f(x)
            # 通过使用 |x| 作为实际输入实现
            pass  # 需要在输入层处理
        
        return output


class SymmetricMLP(nn.Module):
    """
    对称 MLP：保证 f(-x) = f(x)（偶函数）
    
    物理应用：势能函数、概率密度等
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 |x| 保证偶函数性质
        x_abs = torch.abs(x)
        return self.network(x_abs)


def demo_physics_mlp():
    """演示物理约束 MLP"""
    print_section("5. 物理约束 MLP")
    
    x = torch.randn(8, 10)
    
    # 不同约束
    constraints = ['none', 'positive', 'normalized', 'unit_norm', 'bounded']
    
    for constraint in constraints:
        model = PhysicsConstrainedMLP(
            input_dim=10,
            hidden_dims=[32, 32],
            output_dim=5,
            constraint=constraint
        )
        output = model(x)
        
        if constraint == 'positive':
            info = f"最小值={output.min().item():.4f}"
        elif constraint == 'normalized':
            info = f"行和={output.sum(dim=-1)[0].item():.4f}"
        elif constraint == 'unit_norm':
            info = f"范数={output.norm(dim=-1)[0].item():.4f}"
        elif constraint == 'bounded':
            info = f"范围=[{output.min().item():.4f}, {output.max().item():.4f}]"
        else:
            info = f"范围=[{output.min().item():.4f}, {output.max().item():.4f}]"
        
        print(f"约束={constraint:12} 输出形状={output.shape} {info}")
    
    # 对称 MLP 演示
    print("\n对称 MLP (偶函数):")
    sym_model = SymmetricMLP(input_dim=5, hidden_dims=[32], output_dim=1)
    
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    neg_x = -x
    
    y = sym_model(x)
    y_neg = sym_model(neg_x)
    
    print(f"  f(x) = {y.item():.6f}")
    print(f"  f(-x) = {y_neg.item():.6f}")
    print(f"  差异 = {abs(y.item() - y_neg.item()):.2e}")


# ============================================================
# 6. 自适应 MLP
# ============================================================

class AdaptiveMLP(nn.Module):
    """
    自适应 MLP：可以在运行时调整行为
    
    特性：
    - 可选的跳跃连接
    - 动态 dropout
    - 层选择
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # 隐藏层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            if i < len(hidden_dims) - 1:
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.norms.append(nn.LayerNorm(hidden_dims[i+1]))
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        
        # 可学习的跳跃连接权重
        self.skip_weights = nn.Parameter(torch.zeros(len(hidden_dims)))
        
        self.activation = nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        use_skip: bool = True,
        dropout_rate: float = 0.0,
        active_layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        
        x = self.input_proj(x)
        
        skip_features = [x]
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # 检查是否使用该层
            if active_layers is not None and i not in active_layers:
                continue
            
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            
            # 动态 dropout
            if dropout_rate > 0 and self.training:
                x = F.dropout(x, p=dropout_rate)
            
            skip_features.append(x)
        
        # 可选的加权跳跃连接
        if use_skip and len(skip_features) > 1:
            weights = torch.softmax(self.skip_weights[:len(skip_features)], dim=0)
            # 只对形状匹配的特征求和
            # 这里简化处理，只使用最后的特征
            pass
        
        return self.output_proj(x)


def demo_adaptive_mlp():
    """演示自适应 MLP"""
    print_section("6. 自适应 MLP")
    
    model = AdaptiveMLP(
        input_dim=10,
        hidden_dims=[64, 64, 64],
        output_dim=5
    )
    
    x = torch.randn(8, 10)
    
    # 不同配置
    configs = [
        {'use_skip': False, 'dropout_rate': 0.0},
        {'use_skip': True, 'dropout_rate': 0.0},
        {'use_skip': True, 'dropout_rate': 0.3},
        {'use_skip': True, 'dropout_rate': 0.0, 'active_layers': [0, 2]},
    ]
    
    print("不同配置的输出:")
    for config in configs:
        model.train()  # 确保 dropout 生效
        y = model(x, **config)
        print(f"  配置={config} -> 输出均值={y.mean().item():.4f}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" 自定义 MLP 示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # 运行所有演示
    demo_basic_mlp()
    demo_configurable_mlp()
    demo_residual_mlp()
    demo_multihead_mlp()
    demo_physics_mlp()
    demo_adaptive_mlp()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 使用 nn.ModuleList 动态构建层")
    print("  2. 残差连接帮助梯度流动")
    print("  3. 物理约束可以通过输出变换实现")
    print("  4. 多任务学习使用共享主干 + 多个头")
    print("  5. 网络可以设计为在运行时自适应")
    print()


if __name__ == "__main__":
    main()

