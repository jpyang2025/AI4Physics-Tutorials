#!/usr/bin/env python3
"""
使用预训练模型进行图像分类

本脚本演示如何使用 torchvision 的预训练模型进行图像分类，
包括模型加载、图像预处理和推理。

运行方式：
    python image_classification.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import urllib.request
from PIL import Image
import io
import time
from typing import List, Tuple


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. 加载预训练模型
# ============================================================

def load_pretrained_model():
    """加载预训练模型并展示其结构"""
    print_section("1. 加载预训练模型")
    
    # 加载 ResNet-50 预训练模型
    print("加载 ResNet-50 (ImageNet 预训练)...")
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 查看模型结构的顶层模块
    print("\n模型结构 (顶层模块):")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name:15} {module.__class__.__name__:20} 参数: {param_count:,}")
    
    # 获取分类头信息
    print(f"\n分类头 (fc): {model.fc}")
    print(f"输出类别数: {model.fc.out_features}")
    
    return model, weights


# ============================================================
# 2. 图像预处理
# ============================================================

def get_preprocessing_pipeline(weights):
    """获取预处理管道"""
    print_section("2. 图像预处理")
    
    # 方法1：使用权重自带的预处理（推荐）
    preprocess_auto = weights.transforms()
    print("自动预处理流程:")
    print(f"  {preprocess_auto}")
    
    # 方法2：手动定义预处理
    preprocess_manual = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("\n手动预处理流程:")
    print("  1. Resize: 短边缩放到 256")
    print("  2. CenterCrop: 中心裁剪 224x224")
    print("  3. ToTensor: 转为张量 [0, 1]")
    print("  4. Normalize: ImageNet 均值/标准差归一化")
    
    return preprocess_auto


# ============================================================
# 3. 创建测试图像
# ============================================================

def create_test_images():
    """创建测试用的合成图像"""
    print_section("3. 创建测试图像")
    
    # 创建一些简单的合成图像用于演示
    images = {}
    
    # 随机噪声图像
    import numpy as np
    np.random.seed(42)
    
    # 图像1：随机噪声
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    images['random_noise'] = Image.fromarray(noise)
    
    # 图像2：红色渐变
    red_gradient = np.zeros((256, 256, 3), dtype=np.uint8)
    red_gradient[:, :, 0] = np.linspace(0, 255, 256).astype(np.uint8)
    images['red_gradient'] = Image.fromarray(red_gradient)
    
    # 图像3：棋盘格
    checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                checkerboard[i*32:(i+1)*32, j*32:(j+1)*32] = 255
    images['checkerboard'] = Image.fromarray(checkerboard)
    
    print(f"创建了 {len(images)} 张测试图像:")
    for name, img in images.items():
        print(f"  {name}: {img.size}")
    
    return images


# ============================================================
# 4. 单张图像分类
# ============================================================

def classify_single_image(model, preprocess, image, categories, device='cpu'):
    """对单张图像进行分类"""
    model = model.to(device)
    model.eval()
    
    # 预处理
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
    
    # 获取 Top-5
    top5_probs, top5_indices = probs.topk(5, dim=1)
    
    results = []
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        results.append((categories[idx], prob.item()))
    
    return results


def demo_single_image_classification(model, preprocess, weights, images):
    """演示单张图像分类"""
    print_section("4. 单张图像分类")
    
    categories = weights.meta["categories"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    for name, image in images.items():
        print(f"\n图像: {name}")
        results = classify_single_image(model, preprocess, image, categories, device)
        print("  Top-5 预测:")
        for cls, prob in results:
            print(f"    {cls}: {prob:.4f} ({prob*100:.2f}%)")


# ============================================================
# 5. 批量图像分类
# ============================================================

def batch_classify(model, preprocess, images: List[Image.Image], 
                   categories, device='cpu') -> List[Tuple[str, float]]:
    """批量分类图像"""
    model = model.to(device)
    model.eval()
    
    # 预处理所有图像
    batch = torch.stack([preprocess(img) for img in images]).to(device)
    
    # 批量推理
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
    
    results = []
    for pred, conf in zip(predictions, confidences):
        results.append((categories[pred], conf.item()))
    
    return results


def demo_batch_classification(model, preprocess, weights, images):
    """演示批量图像分类"""
    print_section("5. 批量图像分类")
    
    categories = weights.meta["categories"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_list = list(images.values())
    image_names = list(images.keys())
    
    results = batch_classify(model, preprocess, image_list, categories, device)
    
    print("批量分类结果:")
    for name, (cls, conf) in zip(image_names, results):
        print(f"  {name}: {cls} ({conf:.2%})")


# ============================================================
# 6. 推理速度测试
# ============================================================

def benchmark_inference(model, device, batch_sizes=[1, 8, 16, 32], num_iterations=50):
    """基准测试推理速度"""
    print_section("6. 推理速度测试")
    
    model = model.to(device)
    model.eval()
    
    print(f"设备: {device}")
    print(f"迭代次数: {num_iterations}")
    print()
    
    results = []
    
    for batch_size in batch_sizes:
        # 创建随机输入
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        throughput = batch_size * num_iterations / elapsed
        latency = elapsed / num_iterations * 1000  # ms
        
        results.append({
            'batch_size': batch_size,
            'throughput': throughput,
            'latency': latency
        })
        
        print(f"Batch Size: {batch_size:3d} | "
              f"吞吐量: {throughput:7.1f} img/s | "
              f"延迟: {latency:6.2f} ms/batch")
    
    return results


# ============================================================
# 7. 模型对比
# ============================================================

def compare_models():
    """对比不同预训练模型"""
    print_section("7. 模型对比")
    
    model_configs = [
        ('ResNet-18', models.resnet18, 'IMAGENET1K_V1'),
        ('ResNet-50', models.resnet50, 'IMAGENET1K_V1'),
        ('EfficientNet-B0', models.efficientnet_b0, 'IMAGENET1K_V1'),
    ]
    
    print(f"{'模型':<20} {'参数量':>15} {'Top-1 准确率':>15}")
    print("-" * 55)
    
    for name, model_fn, weights_name in model_configs:
        try:
            model = model_fn(weights=weights_name)
            params = sum(p.numel() for p in model.parameters())
            
            # 获取官方报告的准确率
            weights_enum = getattr(models, f"{name.replace('-', '_').replace(' ', '')}_Weights", None)
            if weights_enum and hasattr(weights_enum, weights_name):
                acc = getattr(weights_enum, weights_name).meta.get('acc@1', 'N/A')
            else:
                acc = 'N/A'
            
            print(f"{name:<20} {params:>15,} {str(acc):>15}")
        except Exception as e:
            print(f"{name:<20} 加载失败: {e}")


# ============================================================
# 8. 模型保存与加载
# ============================================================

def demo_save_load():
    """演示模型保存与加载"""
    print_section("8. 模型保存与加载")
    
    # 加载模型
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # 方法1：只保存权重（推荐）
    print("方法1：保存权重 (state_dict)")
    torch.save(model.state_dict(), 'resnet18_weights.pth')
    print("  保存: resnet18_weights.pth")
    
    # 加载权重
    model_loaded = models.resnet18()  # 先创建结构
    model_loaded.load_state_dict(torch.load('resnet18_weights.pth', weights_only=True))
    print("  加载成功!")
    
    # 验证加载正确
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y1 = model(x)
        y2 = model_loaded(x)
    print(f"  验证: 输出差异 = {(y1 - y2).abs().max().item():.2e}")
    
    # 清理
    import os
    os.remove('resnet18_weights.pth')
    print("  清理临时文件")


# ============================================================
# 9. GPU vs CPU 对比
# ============================================================

def compare_cpu_gpu():
    """对比 CPU 和 GPU 推理速度"""
    print_section("9. CPU vs GPU 对比")
    
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # CPU 测试
    print("CPU 推理:")
    cpu_results = benchmark_inference(
        model.to('cpu'), 
        torch.device('cpu'), 
        batch_sizes=[1, 8], 
        num_iterations=10
    )
    
    # GPU 测试（如果可用）
    if torch.cuda.is_available():
        print("\nGPU 推理:")
        gpu_results = benchmark_inference(
            model.to('cuda'), 
            torch.device('cuda'), 
            batch_sizes=[1, 8, 32], 
            num_iterations=50
        )
        
        # 对比
        print("\n加速比 (GPU/CPU):")
        for cpu_r, gpu_r in zip(cpu_results, gpu_results[:len(cpu_results)]):
            speedup = gpu_r['throughput'] / cpu_r['throughput']
            print(f"  Batch Size {cpu_r['batch_size']}: {speedup:.1f}x")
    else:
        print("\nGPU 不可用，跳过 GPU 测试")


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*60)
    print(" PyTorch 预训练模型图像分类示例")
    print("="*60)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    
    # 运行所有演示
    model, weights = load_pretrained_model()
    preprocess = get_preprocessing_pipeline(weights)
    images = create_test_images()
    
    demo_single_image_classification(model, preprocess, weights, images)
    demo_batch_classification(model, preprocess, weights, images)
    
    # 速度测试（使用较小的模型以节省时间）
    model_small = models.resnet18(weights='IMAGENET1K_V1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_inference(model_small, device, batch_sizes=[1, 8, 16], num_iterations=20)
    
    compare_models()
    demo_save_load()
    
    # CPU vs GPU 对比（可能较慢）
    # compare_cpu_gpu()
    
    print("\n" + "="*60)
    print(" 示例运行完成！")
    print("="*60)
    print("\n核心要点:")
    print("  1. 使用 weights 参数加载预训练权重")
    print("  2. 推理前调用 model.eval() 和 torch.no_grad()")
    print("  3. 使用 weights.transforms() 获取预处理流程")
    print("  4. 批量推理比逐个推理更高效")
    print("  5. GPU 推理显著快于 CPU")
    print()


if __name__ == "__main__":
    main()

