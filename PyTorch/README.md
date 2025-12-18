# PyTorch 神经网络入门教程

## 面向物理学研究者的深度学习指南

本教程专为具有 Python 编程基础但不熟悉神经网络的物理科研人员设计。教程采用物理学视角讲解深度学习概念，通过物理类比帮助理解核心原理。

---

## 📚 教程目录

### 第一章：PyTorch 基础
[📁 01_pytorch_basics](./01_pytorch_basics/)

- 张量基础：创建、数据类型、设备管理
- 张量操作：索引、形状变换、数学运算
- 自动微分：计算图、反向传播、高阶导数

### 第二章：神经网络基础
[📁 02_neural_network_fundamentals](./02_neural_network_fundamentals/)

- 感知机与线性模型
- 激活函数（含玻尔兹曼分布视角）
- 前馈神经网络（含通用逼近定理）
- 损失函数（含能量/自由能视角）

### 第三章：使用预训练模型
[📁 03_using_pretrained_models](./03_using_pretrained_models/)

- torchvision 模型库
- 模型推理与导出
- 迁移学习（含微扰理论类比）

### 第四章：构建自定义网络
[📁 04_building_custom_networks](./04_building_custom_networks/)

- nn.Module 详解
- 常用层类型
- 模型设计模式

### 第五章：训练神经网络
[📁 05_training_neural_networks](./05_training_neural_networks/)

- 数据加载与预处理
- 优化器（含动力学系统类比）
- 训练循环
- 验证与测试

### 第六章：进阶技术
[📁 06_advanced_techniques](./06_advanced_techniques/)

- 正则化技术
- 学习率调度（含模拟退火类比）
- 模型保存与加载
- 分布式训练

### 第七章：物理学应用案例
[📁 07_physics_applications](./07_physics_applications/)

- 物理信息神经网络 (PINN)
- 分子动力学应用
- 量子系统模拟

### 附录
[📁 appendix](./appendix/)

- A. 安装与环境配置
- B. 调试技巧
- C. 最佳实践
- D. 资源推荐

---

## 🎯 学习路径

### 快速入门（2-3小时）
1. 第一章：张量基础
2. 第二章：感知机与激活函数
3. 第五章：训练循环基础

### 标准路径（1-2周）
按顺序学习第一章到第六章

### 物理应用导向（3-5天）
1. 第一章：PyTorch 基础
2. 第二章：神经网络基础
3. 第五章：训练神经网络
4. 第七章：物理学应用案例

---

## 🔧 环境要求

```bash
# Python 3.8+
pip install torch torchvision torchaudio

# 可选：CUDA 支持（需要 NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 验证安装

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
```

---

## 📊 物理学类比速查

| 机器学习概念 | 物理学类比 |
|-------------|-----------|
| 损失函数 | 势能/自由能 |
| 梯度下降 | 势能面上的阻尼运动 |
| Softmax | 玻尔兹曼分布 |
| 正则化 | 势能约束 |
| 学习率退火 | 模拟退火 |
| 残差连接 | 微扰理论 |
| BatchNorm | 重整化群变换 |
| 通用逼近定理 | 傅里叶级数展开 |
| 迁移学习 | 已知系统的微扰 |
| 分布式训练 | 空间区域分解 |

---

## 📝 使用说明

1. 每章包含 `README.md` 作为导航
2. 概念讲解在 `.md` 文件中
3. 可运行代码在 `examples/` 目录
4. 建议边读边运行示例代码

---

## 📖 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Deep Learning (Goodfellow et al.)](https://www.deeplearningbook.org/)

---

*本教程持续更新中*

