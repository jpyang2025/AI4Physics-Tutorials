# 第4章：构建自定义网络

## 📖 章节概述

虽然预训练模型非常强大，但在很多场景下，我们需要根据问题的特点设计自己的网络架构。本章将深入介绍如何使用 PyTorch 的 `nn.Module` 构建自定义神经网络。

作为物理科研人员，你可能需要设计特殊的网络结构来：
- 嵌入物理约束（如对称性、守恒律）
- 处理特定格式的数据（如时间序列、图结构）
- 实现新的研究想法

## 🎯 学习目标

完成本章学习后，你将能够：

1. 深入理解 `nn.Module` 的工作原理
2. 掌握各类常用层（全连接、卷积、循环、注意力等）
3. 设计灵活的网络架构
4. 实现残差连接、跳跃连接等高级模式
5. 根据问题需求设计定制化网络

## 📚 章节内容

| 节 | 文件 | 主题 | 预计时间 |
|---|------|------|---------|
| 4.1 | [01_nn_module.md](./01_nn_module.md) | nn.Module 详解 | 45 分钟 |
| 4.2 | [02_layer_types.md](./02_layer_types.md) | 常用层类型 | 50 分钟 |
| 4.3 | [03_model_design_patterns.md](./03_model_design_patterns.md) | 模型设计模式 | 50 分钟 |

## 💻 示例代码

所有可运行的示例代码位于 `examples/` 目录：

- [`custom_mlp.py`](./examples/custom_mlp.py) - 自定义多层感知机
- [`custom_cnn.py`](./examples/custom_cnn.py) - 自定义卷积神经网络
- [`residual_network.py`](./examples/residual_network.py) - 残差网络实现

## 🔬 物理视角：网络架构设计

### 对称性与等变性

物理系统通常具有对称性（平移、旋转、反射等）。我们可以设计网络来尊重这些对称性：

| 对称性 | 网络结构 | 物理应用 |
|--------|---------|---------|
| 平移不变性 | 卷积神经网络 (CNN) | 图像处理、晶格系统 |
| 旋转等变性 | 群等变网络 | 分子动力学、粒子物理 |
| 排列不变性 | 集合网络、GNN | 多体系统、点云 |
| 时间平移 | 循环网络、Transformer | 时间序列、动力学 |

### 架构设计原则

1. **归纳偏置**：将已知的物理知识编码到网络结构中
2. **层次化表示**：类似于重整化群，逐层提取不同尺度的特征
3. **残差学习**：类似于微扰论，学习相对于基准解的修正

## 📋 前置要求

- 完成第1-3章的学习
- 理解神经网络的基本概念
- 熟悉面向对象编程

## 🚀 快速开始

```bash
# 运行示例
cd 04_building_custom_networks/examples
python custom_mlp.py
python custom_cnn.py
python residual_network.py
```

## 🏗️ 网络组件速览

```
nn.Module           # 所有网络的基类
├── 容器类
│   ├── nn.Sequential     # 顺序容器
│   ├── nn.ModuleList     # 模块列表
│   └── nn.ModuleDict     # 模块字典
├── 线性层
│   └── nn.Linear         # 全连接层
├── 卷积层
│   ├── nn.Conv1d/2d/3d   # 卷积层
│   └── nn.ConvTranspose  # 反卷积层
├── 池化层
│   ├── nn.MaxPool        # 最大池化
│   └── nn.AvgPool        # 平均池化
├── 归一化层
│   ├── nn.BatchNorm      # 批归一化
│   └── nn.LayerNorm      # 层归一化
├── 循环层
│   ├── nn.RNN            # 循环神经网络
│   ├── nn.LSTM           # 长短期记忆
│   └── nn.GRU            # 门控循环单元
├── 注意力层
│   └── nn.MultiheadAttention  # 多头注意力
└── 正则化层
    └── nn.Dropout        # Dropout
```

## ⏭️ 下一章预告

掌握了网络构建后，第5章将介绍完整的模型训练流程，包括数据加载、优化器选择和训练循环。

---

*预计总学习时间：约 2.5 小时*

