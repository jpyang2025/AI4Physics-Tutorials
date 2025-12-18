# 5.4 验证与测试

## 📖 概述

验证和测试是模型开发中不可或缺的环节。验证用于调整超参数，测试用于评估最终性能。本节介绍如何正确评估模型性能并避免常见陷阱。

## 🎯 学习目标

- 理解验证集和测试集的区别
- 掌握常用评估指标
- 实现正确的评估流程
- 避免数据泄露

---

## 5.4.1 数据划分策略

### 训练/验证/测试划分

```
全部数据
├── 训练集 (60-80%)
│   └── 用于训练模型参数
├── 验证集 (10-20%)
│   └── 用于调整超参数和早停
└── 测试集 (10-20%)
    └── 用于最终性能评估（只使用一次！）
```

```python
from torch.utils.data import random_split

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    划分数据集
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_set, val_set, test_set
    """
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_set)}")
    print(f"  验证集: {len(val_set)}")
    print(f"  测试集: {len(test_set)}")
    
    return train_set, val_set, test_set
```

### 交叉验证

当数据量有限时，使用 K 折交叉验证。

```python
from sklearn.model_selection import KFold
import numpy as np

def k_fold_cross_validation(dataset, model_fn, train_fn, k=5, seed=42):
    """
    K 折交叉验证
    
    Args:
        dataset: 完整数据集
        model_fn: 创建模型的函数
        train_fn: 训练函数
        k: 折数
        seed: 随机种子
    
    Returns:
        每折的验证结果
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n===== Fold {fold + 1}/{k} =====")
        
        # 创建子集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # 创建新模型
        model = model_fn()
        
        # 训练并评估
        result = train_fn(model, train_subset, val_subset)
        fold_results.append(result)
        
        print(f"Fold {fold + 1} 验证结果: {result}")
    
    # 统计结果
    mean_result = np.mean(fold_results)
    std_result = np.std(fold_results)
    print(f"\n交叉验证结果: {mean_result:.4f} ± {std_result:.4f}")
    
    return fold_results
```

### 时间序列数据划分

对于时间序列数据，不能随机划分！

```python
def time_series_split(dataset, n_splits=5):
    """
    时间序列交叉验证
    
    保持时间顺序：总是用过去预测未来
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    for train_idx, val_idx in tscv.split(range(len(dataset))):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        splits.append((train_subset, val_subset))
        
        print(f"Train: {min(train_idx)}-{max(train_idx)}, "
              f"Val: {min(val_idx)}-{max(val_idx)}")
    
    return splits
```

---

## 5.4.2 评估模式

### train() vs eval()

```python
# 训练模式
model.train()
# - Dropout 层激活，随机丢弃神经元
# - BatchNorm 使用当前批次的均值和方差

# 评估模式
model.eval()
# - Dropout 层关闭，使用所有神经元
# - BatchNorm 使用训练时累积的均值和方差

# 正确的评估流程
def evaluate(model, test_loader, criterion, device):
    model.eval()  # 切换到评估模式
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 关闭梯度计算
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

### torch.no_grad() vs torch.inference_mode()

```python
# no_grad: 关闭梯度追踪
with torch.no_grad():
    output = model(input)
    # 仍然可以对 output 进行需要梯度的操作（在 with 块外）

# inference_mode: 更彻底的推理模式（PyTorch 1.9+）
with torch.inference_mode():
    output = model(input)
    # 完全禁用 autograd，更快更省内存
    # output 张量不能用于后续的梯度计算
```

---

## 5.4.3 分类评估指标

### 基本指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def compute_classification_metrics(y_true, y_pred, num_classes):
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
    """
    metrics = {}
    
    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 精确率、召回率、F1
    if num_classes == 2:
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
    else:
        # 多分类使用宏平均或加权平均
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics
```

### 混淆矩阵可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6)):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.show()
```

### ROC 曲线和 AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

def compute_roc_auc(y_true, y_scores, num_classes):
    """
    计算 ROC 曲线和 AUC
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率/分数
        num_classes: 类别数
    """
    if num_classes == 2:
        # 二分类
        fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    else:
        # 多分类：One-vs-Rest
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}


def plot_roc_curve(roc_data, num_classes):
    """绘制 ROC 曲线"""
    plt.figure(figsize=(8, 6))
    
    if num_classes == 2:
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f"AUC = {roc_data['auc']:.3f}")
    else:
        for i in range(num_classes):
            plt.plot(roc_data['fpr'][i], roc_data['tpr'][i],
                    label=f"Class {i} (AUC = {roc_data['auc'][i]:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

---

## 5.4.4 回归评估指标

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    计算回归指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    """
    metrics = {}
    
    # MSE (均方误差)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # RMSE (均方根误差)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # MAE (平均绝对误差)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R² (决定系数)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # 相对误差
    relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    metrics['mape'] = np.mean(relative_error) * 100  # 平均百分比误差
    
    return metrics


def plot_regression_results(y_true, y_pred, title='回归结果'):
    """
    绘制回归结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 预测值 vs 真实值
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', label='理想')
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title('预测值 vs 真实值')
    axes[0].legend()
    
    # 残差分布
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('残差')
    axes[1].set_ylabel('频数')
    axes[1].set_title('残差分布')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

---

## 5.4.5 物理问题评估指标

### 守恒量误差

```python
def evaluate_conservation_laws(model, test_loader, device):
    """
    评估物理守恒律
    
    检验模型预测是否满足能量守恒、动量守恒等
    """
    model.eval()
    
    energy_errors = []
    momentum_errors = []
    
    with torch.no_grad():
        for initial_state, final_state in test_loader:
            initial_state = initial_state.to(device)
            
            # 模型预测最终状态
            predicted_final = model(initial_state)
            
            # 计算能量（假设 state = [q, p]，H = p²/2 + V(q)）
            q_init, p_init = initial_state[:, :3], initial_state[:, 3:]
            q_pred, p_pred = predicted_final[:, :3], predicted_final[:, 3:]
            
            E_init = 0.5 * (p_init**2).sum(dim=1)  # 动能
            E_pred = 0.5 * (p_pred**2).sum(dim=1)
            
            energy_error = (E_pred - E_init).abs()
            energy_errors.extend(energy_error.cpu().numpy())
            
            # 计算动量
            p_total_init = p_init.sum(dim=1)
            p_total_pred = p_pred.sum(dim=1)
            
            momentum_error = (p_total_pred - p_total_init).abs()
            momentum_errors.extend(momentum_error.cpu().numpy())
    
    metrics = {
        'energy_error_mean': np.mean(energy_errors),
        'energy_error_std': np.std(energy_errors),
        'momentum_error_mean': np.mean(momentum_errors),
        'momentum_error_std': np.std(momentum_errors)
    }
    
    return metrics
```

### 微分方程残差

```python
def evaluate_pde_residual(model, domain_points, boundary_points, 
                          pde_residual_fn, boundary_fn, device):
    """
    评估 PDE 求解器的残差
    
    Args:
        model: PINN 模型
        domain_points: 内部点
        boundary_points: 边界点
        pde_residual_fn: PDE 残差函数
        boundary_fn: 边界条件函数
    """
    model.eval()
    
    # 内部残差
    domain_points = domain_points.to(device)
    domain_points.requires_grad = True
    
    u = model(domain_points)
    residual = pde_residual_fn(u, domain_points)
    
    pde_residual = (residual**2).mean().item()
    
    # 边界残差
    boundary_points = boundary_points.to(device)
    u_boundary = model(boundary_points)
    u_exact = boundary_fn(boundary_points)
    
    boundary_residual = ((u_boundary - u_exact)**2).mean().item()
    
    return {
        'pde_residual': pde_residual,
        'boundary_residual': boundary_residual,
        'total_residual': pde_residual + boundary_residual
    }
```

---

## 5.4.6 完整测试流程

```python
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def predict(self, dataloader):
        """获取预测结果"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_probs.append(probs.cpu())
                else:
                    preds = outputs
                    all_probs = None
                
                all_preds.append(preds.cpu())
                all_targets.append(targets)
        
        result = {
            'predictions': torch.cat(all_preds).numpy(),
            'targets': torch.cat(all_targets).numpy()
        }
        
        if all_probs is not None:
            result['probabilities'] = torch.cat(all_probs).numpy()
        
        return result
    
    def evaluate_classification(self, dataloader, class_names=None):
        """分类评估"""
        result = self.predict(dataloader)
        
        y_true = result['targets']
        y_pred = result['predictions']
        
        num_classes = len(np.unique(y_true))
        
        # 基本指标
        metrics = compute_classification_metrics(y_true, y_pred, num_classes)
        
        # ROC-AUC（如果有概率输出）
        if 'probabilities' in result:
            roc_data = compute_roc_auc(y_true, result['probabilities'], num_classes)
            metrics['roc_auc'] = roc_data
        
        # 打印报告
        print("\n分类评估报告")
        print("=" * 50)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1 分数: {metrics['f1']:.4f}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        
        # 绘制 ROC 曲线
        if 'roc_auc' in metrics:
            plot_roc_curve(metrics['roc_auc'], num_classes)
        
        return metrics
    
    def evaluate_regression(self, dataloader):
        """回归评估"""
        result = self.predict(dataloader)
        
        y_true = result['targets'].flatten()
        y_pred = result['predictions'].flatten()
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # 打印报告
        print("\n回归评估报告")
        print("=" * 50)
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        # 绘制结果
        plot_regression_results(y_true, y_pred)
        
        return metrics


# 使用示例
evaluator = ModelEvaluator(model, device)

# 分类评估
class_names = ['类别0', '类别1', '类别2']
metrics = evaluator.evaluate_classification(test_loader, class_names)

# 回归评估
# metrics = evaluator.evaluate_regression(test_loader)
```

---

## 5.4.7 避免常见陷阱

### 数据泄露

```python
# ❌ 错误：在划分前进行标准化
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # 使用了全部数据的信息
X_train, X_test = train_test_split(X_normalized)

# ✓ 正确：只在训练集上拟合标准化
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 只在训练集上拟合
X_test = scaler.transform(X_test)        # 用训练集的参数变换测试集
```

### 过度调优

```python
# ❌ 错误：反复在测试集上调整模型
for _ in range(100):
    train(model)
    test_acc = evaluate(model, test_loader)  # 测试集信息泄露！
    if test_acc > best:
        adjust_hyperparameters()

# ✓ 正确：用验证集调优，测试集只用一次
for _ in range(100):
    train(model)
    val_acc = evaluate(model, val_loader)  # 用验证集
    if val_acc > best:
        adjust_hyperparameters()

# 最终评估
final_test_acc = evaluate(best_model, test_loader)  # 测试集只用一次
```

### 忘记 eval() 模式

```python
# ❌ 错误：评估时忘记切换模式
def evaluate_wrong(model, test_loader):
    # model.eval() 缺失！
    # Dropout 仍然激活，BatchNorm 使用错误的统计量
    for inputs, targets in test_loader:
        outputs = model(inputs)
        # ...

# ✓ 正确
def evaluate_correct(model, test_loader):
    model.eval()  # 必须！
    with torch.no_grad():  # 推荐
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # ...
```

---

## 🔬 物理视角总结

### 评估的统计力学意义

模型评估可以类比于物理测量：

| 评估概念 | 物理类比 |
|---------|---------|
| 训练集 | 用于确定系统参数的测量 |
| 验证集 | 用于调整实验条件的测量 |
| 测试集 | 独立验证实验 |
| 过拟合 | 过度拟合噪声 |
| 泛化误差 | 系统误差 |

### 不确定性量化

```python
def uncertainty_estimation(model, test_loader, n_samples=30, device='cpu'):
    """
    使用 MC Dropout 估计预测不确定性
    
    类似于物理测量中的误差估计
    """
    model.train()  # 保持 Dropout 激活
    
    all_predictions = []
    
    for _ in range(n_samples):
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu())
        all_predictions.append(torch.cat(predictions))
    
    predictions = torch.stack(all_predictions)  # [n_samples, n_test, ...]
    
    # 均值和标准差
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    return mean, std
```

---

## 📝 练习

1. 实现一个完整的模型评估流程，包含分类指标和可视化
2. 使用交叉验证评估模型性能
3. 实现 MC Dropout 不确定性估计

---

## ⏭️ 下一章预告

掌握了基本训练流程后，第6章将介绍进阶技术，包括正则化、学习率调度、分布式训练等。

