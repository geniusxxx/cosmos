"""
序列打包性能基准测试工具
测量序列打包与传统处理方法的性能差异
"""

import argparse
import time
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.sequence_packing import HAS_XFORMERS

def generate_random_data(batch_size, sizes, device='cuda'):
    """生成随机测试数据
    
    Args:
        batch_size: 批大小
        sizes: 图像尺寸列表 [224, 112, etc.]
        device: 设备
        
    Returns:
        图像列表和文本张量
    """
    # 创建不同尺寸的图像列表
    images = []
    for size in sizes:
        for _ in range(batch_size):
            img = torch.randn(3, size, size, device=device)
            images.append(img)
    
    # 创建文本输入
    texts = torch.randint(0, 49408, (batch_size, 77), device=device)
    
    return images, texts

def benchmark_model(model, images, texts, n_repeat=10, warm_up=3):
    """基准测试模型性能
    
    Args:
        model: 要测试的模型
        images: 图像输入
        texts: 文本输入
        n_repeat: 重复次数
        warm_up: 预热次数
        
    Returns:
        平均执行时间(ms)
    """
    # 预热
    for _ in range(warm_up):
        with torch.no_grad():
            _ = model(images, texts)
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_repeat):
        with torch.no_grad():
            _ = model(images, texts)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_repeat * 1000  # 转换为毫秒
    return avg_time

def run_benchmark(model_name, batch_sizes, n_repeat=10):
    """运行完整的基准测试
    
    Args:
        model_name: 模型名称
        batch_sizes: 要测试的批大小列表
        n_repeat: 每个配置重复的次数
        
    Returns:
        基准测试结果字典
    """
    results = {
        'batch_size': [],
        'standard': [],
        'sequence_packing': [],
        'speedup': []
    }
    
    # 为每个批大小运行测试
    for batch_size in batch_sizes:
        print(f"\n测试批大小: {batch_size}")
        
        # 创建模型
        model, _, _ = create_model_and_transforms(model_name)
        model = model.cuda().eval()
        
        # 确保模型支持序列打包
        if not hasattr(model, 'use_sequence_packing'):
            print("警告: 模型不支持序列打包，跳过")
            continue
        
        # 生成测试数据 - 两种全局裁剪(224x224)和六种局部裁剪(96x96)
        image_sizes = [224, 224, 96, 96, 96, 96, 96, 96]
        images, texts = generate_random_data(batch_size, image_sizes)
        
        # 测试标准处理
        model.use_sequence_packing = False
        standard_time = benchmark_model(model, images, texts, n_repeat)
        print(f"标准处理时间: {standard_time:.2f} ms")
        
        # 测试序列打包
        if HAS_XFORMERS:
            model.use_sequence_packing = True
            packing_time = benchmark_model(model, images, texts, n_repeat)
            print(f"序列打包时间: {packing_time:.2f} ms")
            speedup = (standard_time / packing_time - 1) * 100
            print(f"加速比: {speedup:.1f}%")
        else:
            print("xFormers未安装，无法测试序列打包")
            packing_time = float('nan')
            speedup = float('nan')
        
        # 记录结果
        results['batch_size'].append(batch_size)
        results['standard'].append(standard_time)
        results['sequence_packing'].append(packing_time)
        results['speedup'].append(speedup)
        
        # 释放内存
        del model
        torch.cuda.empty_cache()
    
    return results

def plot_results(results, output_file=None):
    """绘制基准测试结果图表
    
    Args:
        results: 基准测试结果字典
        output_file: 输出文件路径
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制执行时间对比
    plt.subplot(2, 1, 1)
    plt.plot(results['batch_size'], results['standard'], 'o-', label='标准处理')
    plt.plot(results['batch_size'], results['sequence_packing'], 'o-', label='序列打包')
    plt.xlabel('批大小')
    plt.ylabel('执行时间 (ms)')
    plt.title('序列打包性能对比')
    plt.grid(True)
    plt.legend()
    
    # 绘制加速比
    plt.subplot(2, 1, 2)
    plt.bar(results['batch_size'], results['speedup'])
    plt.xlabel('批大小')
    plt.ylabel('加速比 (%)')
    plt.title('序列打包加速率')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file)
        print(f"结果图表已保存到 {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='序列打包性能基准测试')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='模型架构')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32], help='要测试的批大小列表')
    parser.add_argument('--repeats', type=int, default=10, help='每次测试的重复次数')
    parser.add_argument('--output', type=str, default='sequence_packing_benchmark.png', help='输出图表文件')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        return
    
    if not HAS_XFORMERS:
        print("警告: xFormers未安装，无法使用序列打包")
        print("安装命令: pip install xformers>=0.0.20")
    
    print(f"运行序列打包性能基准测试，模型: {args.model}")
    results = run_benchmark(args.model, args.batch_sizes, args.repeats)
    
    # 打印结果表格
    print("\n=== 基准测试结果 ===")
    print("批大小 | 标准处理(ms) | 序列打包(ms) | 加速比(%)")
    print("------|--------------|--------------|----------")
    for i in range(len(results['batch_size'])):
        print(f"{results['batch_size'][i]:6d} | {results['standard'][i]:12.2f} | {results['sequence_packing'][i]:12.2f} | {results['speedup'][i]:10.1f}")
    
    # 绘制图表
    plot_results(results, args.output)

if __name__ == '__main__':
    main() 