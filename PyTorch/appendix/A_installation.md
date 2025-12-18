# 附录 A：安装与环境配置

## 📖 概述

本附录介绍如何安装 PyTorch 及相关工具，配置开发环境，以及解决常见的安装问题。

---

## A.1 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 64位处理器 | 多核心处理器 |
| RAM | 8 GB | 16 GB+ |
| GPU | 无（CPU模式） | NVIDIA GPU (CUDA) |
| 存储 | 5 GB | 20 GB+ SSD |

### 软件要求

| 软件 | 版本要求 |
|------|---------|
| Python | 3.8 - 3.11 |
| pip | 最新版本 |
| CUDA（可选） | 11.8 / 12.1 |
| cuDNN（可选） | 与 CUDA 版本匹配 |

---

## A.2 安装 PyTorch

### 方法一：使用 pip（推荐）

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统的安装命令。

```bash
# CPU 版本
pip install torch torchvision torchaudio

# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方法二：使用 conda

```bash
# 创建新环境
conda create -n pytorch python=3.10

# 激活环境
conda activate pytorch

# CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# CUDA 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 验证安装

```python
import torch

# 基本信息
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    
    # 简单测试
    x = torch.randn(3, 3).cuda()
    print(f"GPU 计算测试: {x @ x.T}")

# 张量操作测试
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"张量相加: {a + b}")
```

---

## A.3 虚拟环境管理

### 使用 venv

```bash
# 创建虚拟环境
python -m venv pytorch_env

# 激活环境
# Linux/macOS
source pytorch_env/bin/activate
# Windows
pytorch_env\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio

# 退出环境
deactivate
```

### 使用 conda

```bash
# 创建环境
conda create -n pytorch python=3.10

# 激活环境
conda activate pytorch

# 查看已有环境
conda env list

# 导出环境
conda env export > environment.yml

# 从文件创建环境
conda env create -f environment.yml

# 删除环境
conda env remove -n pytorch
```

---

## A.4 依赖管理

### requirements.txt

```txt
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# 数据处理
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# 可视化
matplotlib>=3.4.0
seaborn>=0.11.0

# 机器学习工具
scikit-learn>=0.24.0
tqdm>=4.60.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# 可选：图像处理
Pillow>=8.0.0
opencv-python>=4.5.0

# 可选：TensorBoard
tensorboard>=2.5.0
```

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## A.5 CUDA 配置

### 检查 NVIDIA 驱动

```bash
# 检查驱动版本
nvidia-smi
```

### CUDA 版本兼容性

| CUDA 版本 | 最低驱动版本 | PyTorch 支持 |
|-----------|-------------|-------------|
| CUDA 12.1 | 525.60.13+ | ✓ |
| CUDA 11.8 | 520.61.05+ | ✓ |
| CUDA 11.7 | 515.43.04+ | ✓ |

### 多 CUDA 版本管理

```bash
# 查看 CUDA 路径
echo $CUDA_HOME

# 设置特定 CUDA 版本
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## A.6 IDE 配置

### VS Code

推荐扩展：
- Python
- Pylance
- Jupyter
- Python Docstring Generator

`settings.json` 配置：

```json
{
    "python.defaultInterpreterPath": "/path/to/your/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.askForKernelRestart": false
}
```

### PyCharm

1. 设置解释器：`File > Settings > Project > Python Interpreter`
2. 配置代码风格：`File > Settings > Editor > Code Style > Python`
3. 启用科学模式：`View > Scientific Mode`

### Jupyter Notebook

```bash
# 安装
pip install jupyter

# 添加虚拟环境到 Jupyter
python -m ipykernel install --user --name=pytorch --display-name="PyTorch"

# 启动
jupyter notebook
```

---

## A.7 Docker 配置

### 使用官方镜像

```bash
# 拉取镜像
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 运行容器
docker run --gpus all -it pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 挂载本地目录
docker run --gpus all -v $(pwd):/workspace -it pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

### 自定义 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 安装额外依赖
RUN pip install numpy pandas matplotlib scikit-learn tqdm jupyter

# 设置工作目录
WORKDIR /workspace

# 复制代码
COPY . /workspace

# 暴露 Jupyter 端口
EXPOSE 8888

# 启动 Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

---

## A.8 常见安装问题

### 问题 1：CUDA 版本不匹配

**症状**：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**：
```bash
# 检查 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 安装匹配的 PyTorch 版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2：显存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```python
# 减小 batch size
batch_size = 16  # 从 64 减小

# 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 清理缓存
torch.cuda.empty_cache()

# 使用梯度检查点
from torch.utils.checkpoint import checkpoint
```

### 问题 3：pip 安装超时

**解决方案**：
```bash
# 使用国内镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 增加超时时间
pip install torch --timeout 1000
```

### 问题 4：conda 环境冲突

**解决方案**：
```bash
# 创建干净环境
conda create -n pytorch_clean python=3.10

# 只从 pytorch 官方渠道安装
conda install pytorch torchvision -c pytorch --override-channels
```

---

## A.9 性能优化配置

### 环境变量

```bash
# 设置线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 启用 cuDNN 自动调优
export CUDA_VISIBLE_DEVICES=0  # 选择 GPU
```

### PyTorch 配置

```python
# 启用 cuDNN benchmark（固定输入大小时加速）
torch.backends.cudnn.benchmark = True

# 设置默认数据类型
torch.set_default_dtype(torch.float32)

# 设置线程数
torch.set_num_threads(4)

# 启用 TF32（Ampere GPU）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## A.10 更新与卸载

### 更新 PyTorch

```bash
# pip 更新
pip install --upgrade torch torchvision torchaudio

# conda 更新
conda update pytorch torchvision -c pytorch
```

### 卸载

```bash
# pip 卸载
pip uninstall torch torchvision torchaudio

# conda 卸载
conda remove pytorch torchvision
```

---

## 📋 安装检查清单

- [ ] Python 版本正确（3.8-3.11）
- [ ] 虚拟环境已创建并激活
- [ ] PyTorch 安装成功
- [ ] `torch.cuda.is_available()` 返回 True（如果有 GPU）
- [ ] 简单张量运算正常
- [ ] GPU 运算正常（如果有 GPU）
- [ ] torchvision 安装成功
- [ ] 其他依赖安装成功

