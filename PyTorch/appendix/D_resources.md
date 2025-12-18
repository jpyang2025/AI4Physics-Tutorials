# 附录 D：资源推荐

## 📖 概述

本附录汇总了学习 PyTorch 和深度学习的优质资源，包括官方文档、教程、书籍、论文和工具。

---

## D.1 官方资源

### PyTorch 官方

| 资源 | 链接 | 说明 |
|------|------|------|
| 官方文档 | [pytorch.org/docs](https://pytorch.org/docs/) | API 参考和教程 |
| 官方教程 | [pytorch.org/tutorials](https://pytorch.org/tutorials/) | 从入门到进阶 |
| 官方论坛 | [discuss.pytorch.org](https://discuss.pytorch.org/) | 问答社区 |
| GitHub | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | 源代码 |
| 官方博客 | [pytorch.org/blog](https://pytorch.org/blog/) | 最新动态 |

### 相关库

| 库 | 链接 | 用途 |
|---|------|------|
| torchvision | [pytorch.org/vision](https://pytorch.org/vision/) | 计算机视觉 |
| torchaudio | [pytorch.org/audio](https://pytorch.org/audio/) | 音频处理 |
| torchtext | [pytorch.org/text](https://pytorch.org/text/) | 文本处理 |
| PyTorch Geometric | [pyg.org](https://pyg.org/) | 图神经网络 |
| PyTorch Lightning | [lightning.ai](https://lightning.ai/) | 训练框架 |

---

## D.2 在线教程

### 入门教程

| 教程 | 链接 | 特点 |
|------|------|------|
| PyTorch 60 分钟入门 | [官方教程](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 官方快速入门 |
| Learn PyTorch | [learnpytorch.io](https://www.learnpytorch.io/) | 免费完整课程 |
| Made With ML | [madewithml.com](https://madewithml.com/) | MLOps 导向 |
| Full Stack Deep Learning | [fullstackdeeplearning.com](https://fullstackdeeplearning.com/) | 工程实践 |

### 视频课程

| 课程 | 平台 | 说明 |
|------|------|------|
| Deep Learning with PyTorch | freeCodeCamp | 免费完整课程 |
| PyTorch for Deep Learning | Udacity | 免费纳米学位 |
| CS231n | Stanford | 计算机视觉经典 |
| CS224n | Stanford | NLP 经典 |
| Fast.ai | fast.ai | 实践导向 |

### 物理学专题

| 资源 | 链接 | 主题 |
|------|------|------|
| Physics-Informed Neural Networks | [原始论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125) | PINN 基础 |
| DeepMind AlphaFold | [nature.com](https://www.nature.com/articles/s41586-021-03819-2) | 蛋白质结构预测 |
| Neural Network Potentials | [reviews](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01111) | 分子动力学势能面 |
| Machine Learning for Physics | [arxiv](https://arxiv.org/abs/1903.04506) | 物理学机器学习综述 |

---

## D.3 书籍推荐

### 深度学习理论

| 书名 | 作者 | 特点 |
|------|------|------|
| Deep Learning | Goodfellow et al. | 经典教材，理论全面 |
| Dive into Deep Learning | 李沐等 | 交互式，代码丰富 |
| Neural Networks and Deep Learning | Michael Nielsen | 在线免费，直观讲解 |
| Pattern Recognition and Machine Learning | Bishop | 贝叶斯视角 |

### PyTorch 专题

| 书名 | 作者 | 特点 |
|------|------|------|
| Deep Learning with PyTorch | Eli Stevens et al. | 实践导向 |
| Programming PyTorch for Deep Learning | Ian Pointer | 项目驱动 |
| PyTorch Pocket Reference | Joe Papa | 快速参考 |

### 物理学+机器学习

| 书名 | 作者 | 特点 |
|------|------|------|
| Machine Learning for Physics | Mehta et al. | 物理视角 |
| Statistical Mechanics of Learning | Engel & Van den Broeck | 统计力学方法 |
| The Principles of Deep Learning Theory | Roberts et al. | 理论物理方法分析神经网络 |

---

## D.4 论文推荐

### 经典论文

| 论文 | 年份 | 贡献 |
|------|------|------|
| AlexNet | 2012 | CNN 复兴 |
| VGGNet | 2014 | 深层网络 |
| ResNet | 2015 | 残差连接 |
| Batch Normalization | 2015 | 训练加速 |
| Adam Optimizer | 2015 | 自适应学习率 |
| Dropout | 2014 | 正则化 |
| Attention Is All You Need | 2017 | Transformer |

### 物理学应用

| 论文 | 年份 | 主题 |
|------|------|------|
| Physics-Informed Neural Networks | 2019 | PINN 原始论文 |
| SchNet | 2017 | 分子性质预测 |
| DeepMD | 2018 | 分子动力学势能 |
| Neural Quantum States | 2017 | 量子多体系统 |
| FermiNet | 2020 | 费米子波函数 |
| PauliNet | 2020 | 反对称神经网络 |

### 最新进展

建议关注的会议和期刊：
- NeurIPS (Neural Information Processing Systems)
- ICML (International Conference on Machine Learning)
- ICLR (International Conference on Learning Representations)
- Physical Review X
- Nature Machine Intelligence
- Journal of Chemical Physics

---

## D.5 实用工具

### 可视化

| 工具 | 用途 | 链接 |
|------|------|------|
| TensorBoard | 训练监控 | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) |
| Weights & Biases | 实验跟踪 | [wandb.ai](https://wandb.ai/) |
| Netron | 模型可视化 | [netron.app](https://netron.app/) |
| torchviz | 计算图可视化 | [github](https://github.com/szagoruyko/pytorchviz) |

### 性能分析

| 工具 | 用途 | 说明 |
|------|------|------|
| PyTorch Profiler | 性能分析 | 内置工具 |
| NVIDIA Nsight | GPU 分析 | CUDA 性能 |
| torch.utils.benchmark | 基准测试 | 时间测量 |
| memory_profiler | 内存分析 | Python 包 |

### 模型库

| 库 | 用途 | 链接 |
|---|------|------|
| Hugging Face | NLP/CV 模型 | [huggingface.co](https://huggingface.co/) |
| timm | 图像模型 | [github](https://github.com/huggingface/pytorch-image-models) |
| torchvision.models | 预训练模型 | PyTorch 内置 |
| OpenMMLab | 计算机视觉 | [openmmlab.com](https://openmmlab.com/) |

### 分子模拟专用

| 工具 | 用途 | 链接 |
|------|------|------|
| SchNetPack | 分子建模 | [github](https://github.com/atomistic-machine-learning/schnetpack) |
| DeePMD-kit | 分子动力学 | [github](https://github.com/deepmodeling/deepmd-kit) |
| TorchMD | 可微分 MD | [github](https://github.com/torchmd/torchmd) |
| e3nn | 等变神经网络 | [github](https://github.com/e3nn/e3nn) |

---

## D.6 社区资源

### 问答社区

| 平台 | 链接 | 特点 |
|------|------|------|
| PyTorch Forums | [discuss.pytorch.org](https://discuss.pytorch.org/) | 官方论坛 |
| Stack Overflow | [stackoverflow.com/questions/tagged/pytorch](https://stackoverflow.com/questions/tagged/pytorch) | 技术问答 |
| Reddit r/MachineLearning | [reddit.com/r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | 讨论社区 |

### 代码资源

| 平台 | 链接 | 说明 |
|------|------|------|
| Papers With Code | [paperswithcode.com](https://paperswithcode.com/) | 论文+代码 |
| Awesome PyTorch | [github](https://github.com/bharathgs/Awesome-pytorch-list) | 资源列表 |
| PyTorch Hub | [pytorch.org/hub](https://pytorch.org/hub/) | 模型仓库 |

---

## D.7 物理学家的额外资源

### 入门建议

对于物理学背景的研究者，建议学习顺序：

1. **基础**：PyTorch 官方 60 分钟入门
2. **理论**：《Deep Learning》第一部分
3. **实践**：动手实现简单的 PINN
4. **进阶**：阅读相关领域论文

### 数学桥梁

| 物理概念 | 机器学习对应 |
|---------|------------|
| 能量函数 | 损失函数 |
| 配分函数 | 归一化常数 |
| 玻尔兹曼分布 | Softmax |
| 模拟退火 | 学习率调度 |
| 变分原理 | 变分推断 |
| 重整化群 | 深层网络/BatchNorm |
| 对称性 | 等变神经网络 |

### 推荐论文阅读顺序

1. **入门**：PINN 原始论文（2019）
2. **分子模拟**：SchNet（2017），DeepMD（2018）
3. **量子系统**：Neural Quantum States（2017）
4. **前沿**：FermiNet（2020），Equivariant GNN

---

## D.8 持续学习

### 保持更新

- 订阅 PyTorch 官方博客
- 关注 Twitter/X 上的研究者
- 参加 ML/Physics 线上研讨会
- 阅读 arXiv 最新论文

### 实践项目建议

1. **初级**：实现基本的分类/回归模型
2. **中级**：迁移学习应用到自己的数据
3. **高级**：实现 PINN 求解你研究中的 PDE
4. **进阶**：结合领域知识设计新的网络架构

### 贡献社区

- 提交 bug 报告和功能建议
- 回答论坛问题
- 分享你的代码和经验
- 撰写教程和博客

---

## 📌 快速链接收藏

```
# 官方资源
https://pytorch.org/docs/
https://pytorch.org/tutorials/

# 预训练模型
https://huggingface.co/models
https://pytorch.org/hub/

# 论文实现
https://paperswithcode.com/

# 问答
https://discuss.pytorch.org/
https://stackoverflow.com/questions/tagged/pytorch
```

---

*最后更新：2024年*

