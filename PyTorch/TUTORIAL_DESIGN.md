# PyTorch 神经网络入门教程 - 设计文档

## 目标读者

本教程面向具备以下背景的物理科研人员：
- ✅ 熟练的 Python 编程能力
- ✅ 扎实的数学基础（微积分、线性代数、概率统计）
- ✅ 熟悉科学计算库（NumPy、SciPy、Matplotlib）
- ❌ 不熟悉深度学习/神经网络

## 教程特色

1. **从物理直觉出发**：用物理学的视角解释神经网络概念（如能量最小化、梯度下降类比于势能面上的运动）
2. **数学不回避**：充分利用读者的数学背景，给出严谨的数学推导
3. **代码即实践**：每章配备可运行的 Jupyter Notebook 或 Python 脚本
4. **物理相关案例**：使用物理学相关的数据和问题作为示例

---

## 教程目录结构

```
PyTorch/
├── TUTORIAL_DESIGN.md          # 本设计文档
├── README.md                   # 教程总览与使用指南
│
├── 01_pytorch_basics/          # 第1章：PyTorch 基础与张量操作
│   ├── README.md               # 章节说明
│   ├── 01_tensor_basics.md     # 张量基础
│   ├── 02_tensor_operations.md # 张量运算
│   ├── 03_autograd.md          # 自动微分
│   └── examples/               # 示例代码
│       ├── tensor_demo.py
│       └── autograd_demo.py
│
├── 02_neural_network_fundamentals/  # 第2章：神经网络基础理论
│   ├── README.md
│   ├── 01_perceptron.md        # 感知机与线性模型
│   ├── 02_activation_functions.md  # 激活函数
│   ├── 03_feedforward_networks.md  # 前馈神经网络
│   ├── 04_loss_functions.md    # 损失函数
│   └── examples/
│       ├── linear_regression.py
│       └── simple_classifier.py
│
├── 03_using_pretrained_models/     # 第3章：使用预训练模型
│   ├── README.md
│   ├── 01_torchvision_models.md    # torchvision 模型库
│   ├── 02_model_inference.md       # 模型推理
│   ├── 03_transfer_learning.md     # 迁移学习
│   └── examples/
│       ├── image_classification.py
│       └── feature_extraction.py
│
├── 04_building_custom_networks/    # 第4章：构建自定义网络
│   ├── README.md
│   ├── 01_nn_module.md             # nn.Module 详解
│   ├── 02_layer_types.md           # 常用层类型
│   ├── 03_model_design_patterns.md # 模型设计模式
│   └── examples/
│       ├── custom_mlp.py
│       ├── custom_cnn.py
│       └── residual_network.py
│
├── 05_training_neural_networks/    # 第5章：训练神经网络
│   ├── README.md
│   ├── 01_data_loading.md          # 数据加载（Dataset, DataLoader）
│   ├── 02_optimizers.md            # 优化器
│   ├── 03_training_loop.md         # 训练循环
│   ├── 04_validation_testing.md    # 验证与测试
│   └── examples/
│       ├── mnist_training.py
│       └── regression_training.py
│
├── 06_advanced_techniques/         # 第6章：进阶技术
│   ├── README.md
│   ├── 01_regularization.md        # 正则化技术
│   ├── 02_learning_rate_scheduling.md  # 学习率调度
│   ├── 03_model_save_load.md       # 模型保存与加载
│   ├── 04_hyperparameter_tuning.md # 超参数调优
│   ├── 05_gpu_training.md          # GPU 训练与多卡并行
│   └── examples/
│       ├── regularization_demo.py
│       └── checkpoint_demo.py
│
├── 07_physics_applications/        # 第7章：物理学应用案例
│   ├── README.md
│   ├── 01_function_fitting.md      # 复杂函数拟合
│   ├── 02_pinn_basics.md           # 物理信息神经网络 (PINN) 基础
│   ├── 03_pinn_pde_solving.md      # 用 PINN 求解偏微分方程
│   ├── 04_spectral_analysis.md     # 谱分析与信号处理
│   └── examples/
│       ├── harmonic_oscillator.py
│       ├── heat_equation_pinn.py
│       └── schrodinger_equation.py
│
└── appendix/                       # 附录
    ├── A_environment_setup.md      # 环境配置
    ├── B_math_review.md            # 数学回顾
    ├── C_debugging_tips.md         # 调试技巧
    └── D_resources.md              # 进阶资源
```

---

## 各章节详细内容

### 第1章：PyTorch 基础与张量操作

**学习目标**：掌握 PyTorch 的核心数据结构——张量，理解自动微分机制。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 1.1 | 张量基础 | 张量创建、数据类型、设备(CPU/GPU)、与 NumPy 互转 |
| 1.2 | 张量运算 | 索引切片、形状变换、广播机制、数学运算 |
| 1.3 | 自动微分 | `requires_grad`、计算图、`backward()`、梯度累积 |

**物理视角**：将张量理解为多维物理量（如应力张量、电磁场张量），自动微分理解为沿势能面计算力（负梯度）。

---

### 第2章：神经网络基础理论

**学习目标**：理解神经网络的数学原理，建立直觉。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 2.1 | 感知机与线性模型 | 线性回归、逻辑回归、决策边界 |
| 2.2 | 激活函数 | ReLU、Sigmoid、Tanh、物理意义 |
| 2.3 | 前馈神经网络 | 万能近似定理、层与深度 |
| 2.4 | 损失函数 | MSE、交叉熵、物理中的能量函数类比 |

**物理视角**：
- 神经网络训练 ≈ 能量最小化过程
- 损失函数 ≈ 系统的势能/自由能
- 梯度下降 ≈ 阻尼振子在势能面上的运动

---

### 第3章：使用预训练模型

**学习目标**：快速上手使用现有模型，理解迁移学习的价值。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 3.1 | torchvision 模型库 | ResNet、VGG、模型加载 |
| 3.2 | 模型推理 | `eval()` 模式、`torch.no_grad()`、预处理 |
| 3.3 | 迁移学习 | 特征提取、微调、冻结层 |

**实际应用**：使用预训练模型对实验图像（如显微镜图像、光谱图）进行分类或特征提取。

---

### 第4章：构建自定义网络

**学习目标**：能够根据问题需求设计和实现神经网络架构。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 4.1 | nn.Module 详解 | `__init__`、`forward`、参数管理 |
| 4.2 | 常用层类型 | Linear、Conv2d、LSTM、Embedding |
| 4.3 | 模型设计模式 | Sequential、残差连接、注意力机制 |

**物理视角**：网络架构设计类似于选择合适的基函数展开（如傅里叶级数、球谐函数）。

---

### 第5章：训练神经网络

**学习目标**：掌握完整的模型训练流程。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 5.1 | 数据加载 | `Dataset`、`DataLoader`、数据增强 |
| 5.2 | 优化器 | SGD、Adam、动量、学习率 |
| 5.3 | 训练循环 | 前向传播、反向传播、参数更新 |
| 5.4 | 验证与测试 | 训练/验证/测试集划分、过拟合检测 |

**物理视角**：
- 优化器 ≈ 不同的动力学模拟方法（SGD ≈ 过阻尼朗之万动力学，Adam ≈ 自适应步长积分器）
- 批量训练 ≈ 蒙特卡洛采样

---

### 第6章：进阶技术

**学习目标**：提升模型性能和训练效率。

| 节 | 内容 | 关键概念 |
|---|------|---------|
| 6.1 | 正则化技术 | Dropout、BatchNorm、L1/L2 正则 |
| 6.2 | 学习率调度 | StepLR、CosineAnnealing、预热 |
| 6.3 | 模型保存与加载 | `state_dict`、检查点、ONNX 导出 |
| 6.4 | 超参数调优 | 网格搜索、随机搜索、贝叶斯优化 |
| 6.5 | GPU 训练 | CUDA、多卡并行、混合精度 |

---

### 第7章：物理学应用案例

**学习目标**：将神经网络应用于物理问题。

| 节 | 内容 | 示例问题 |
|---|------|---------|
| 7.1 | 复杂函数拟合 | 多变量非线性函数、实验数据拟合 |
| 7.2 | PINN 基础 | 物理约束损失函数、边界条件嵌入 |
| 7.3 | PINN 求解 PDE | 热传导方程、波动方程、N-S 方程 |
| 7.4 | 谱分析 | 信号去噪、频谱预测 |

**重点案例**：
1. **谐振子**：用神经网络拟合谐振子的解
2. **热传导方程**：用 PINN 求解一维热传导
3. **薛定谔方程**：用神经网络求解量子力学问题

---

## 教学方法

### 每章结构
1. **概念讲解**（Markdown）：理论知识与数学推导
2. **代码示例**（Python）：可直接运行的代码
3. **练习题**：巩固理解
4. **延伸阅读**：论文和进阶资料

### 代码规范
- 使用类型注解提高可读性
- 详细的中文注释
- 遵循 PEP 8 风格
- 提供完整的可运行示例

---

## 技术要求

### 软件环境
- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- numpy, matplotlib, scipy
- jupyter（可选）

### 硬件建议
- 基础章节：CPU 即可
- 进阶章节：建议有 CUDA 兼容的 GPU

---

## 开发计划

| 阶段 | 内容 | 预计工作量 |
|------|------|-----------|
| Phase 1 | 第1-2章（基础） | 基础理论 |
| Phase 2 | 第3-5章（核心） | 实践技能 |
| Phase 3 | 第6章（进阶） | 优化技术 |
| Phase 4 | 第7章（应用） | 物理案例 |
| Phase 5 | 附录与完善 | 补充材料 |

---

## 参考资源

1. [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
2. [PyTorch 官方教程](https://pytorch.org/tutorials/)
3. 《Deep Learning》 - Goodfellow, Bengio, Courville
4. 《Physics-Informed Machine Learning》相关论文

---

## 设计原则

1. **渐进式学习**：从简单到复杂，每个新概念都建立在之前的基础上
2. **实践导向**：每个概念都有对应的可运行代码
3. **物理关联**：尽可能用物理学的语言和类比解释概念
4. **模块化设计**：各章节相对独立，可按需学习

---

*设计文档版本：v1.0*  
*创建日期：2024年*

