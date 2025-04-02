#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Evaluation code is borrowed from https://github.com/mlfoundations/datacomp/blob/main/eval_utils/wds_eval.py
Licensed under MIT License, see ACKNOWLEDGEMENTS for details.
"""
import os
import sys
        
import copy
import argparse
import open_clip
import torch
import torch.nn as nn
from timm.models.fastvit import ConvMlp
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from clip_benchmark.metrics import zeroshot_classification as zsc
import onnxruntime as ort
import numpy as np
import torch.nn.functional as F
import json

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     # debugpy.wait_for_client.cancel()
#     debugpy.wait_for_client()
    
# except Exception as e:
#     pass

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args(parser):
    parser.add_argument(
        "--model-arch",
        type=str,
        required=True,
        help="Specify model arch from the available choices.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        nargs='+',
        help="Specify location of model checkpoint. For ONNX models, you can provide either one path (will auto-detect text model) or two paths (containing 'visual' and 'text' in names).",
    )
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=["flickr30k"],
        choices=["flickr30k", "imagenet1k", "mscoco_captions"],
        help="Specify which tasks to evaluate on"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/xuboyu/Data/datasets",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Input resolution for images"
    )
    return parser

def reparameterize_model(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model

class ONNXModelWrapper(nn.Module):
    def __init__(self, visual_path, text_path, model_arch, resolution=256, batch_size=64):
        super().__init__()
        self.visual_path = visual_path
        self.text_path = text_path
        self.model_arch = model_arch
        self.resolution = resolution
        self.batch_size = batch_size
        
        # 初始化预处理和tokenizer
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_arch,
            pretrained=False,
            # image_mean=(0, 0, 0),
            # image_std=(1, 1, 1),
            # image_interpolation="bilinear",
            force_image_size=(resolution, resolution)
        )
        self.tokenizer = open_clip.get_tokenizer(model_arch)
        
        # ONNX Runtime基础配置
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 简单的CUDA Provider配置
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }),
            'CPUExecutionProvider'
        ]
        
        # 初始化视觉会话
        self.visual_session = ort.InferenceSession(
            visual_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # 如果text_path是字符串，则创建ONNX会话；如果是SplitTextEncoder实例，则直接使用
        if isinstance(text_path, str):
            self.text_session = ort.InferenceSession(
                text_path, 
                sess_options=sess_options,
                providers=providers
            )
            self.text_input_name = self.text_session.get_inputs()[0].name
            self.text_output_name = self.text_session.get_outputs()[0].name
        else:  # text_path是SplitTextEncoder实例
            self.text_encoder = text_path
            
        # 获取视觉模型的输入输出名称
        self.visual_input_name = self.visual_session.get_inputs()[0].name
        self.visual_output_name = self.visual_session.get_outputs()[0].name
        
        # 检查是否是动态轴模型
        self.is_dynamic = self._check_dynamic_axes()
        
        # 预热模型
        self._warmup()
    
    def _warmup(self):
        """预热模型以优化性能"""
        print("\n预热模型中...")
        
        # 使用参数化的分辨率和批处理大小
        dummy_image = torch.randn(self.batch_size, 3, self.resolution, self.resolution)
        
        if self.is_dynamic:
            # 动态轴模式：直接处理整个batch
            self.encode_image(dummy_image)
            dummy_text = ["warm up text"] * self.batch_size
            self.encode_text(dummy_text)
        else:
            # 固定轴模式：逐个处理
            print("固定轴模式：逐个预热")
            for i in range(min(2, self.batch_size)):  # 只预热前两个样本即可
                self.encode_image(dummy_image[i:i+1])
            dummy_text = ["warm up text"]
            self.encode_text(dummy_text)
        
        print("模型预热完成")
    
    def _check_dynamic_axes(self):
        """通过文件名检查是否是动态轴模型"""
        is_dynamic = 'dynamic' in self.visual_path.lower()
        print(f"\n模型类型: {'动态轴' if is_dynamic else '固定轴'}")
        return is_dynamic
    
    @torch.no_grad()
    def encode_image(self, image):
        """优化的图像编码"""
        if isinstance(image, torch.Tensor):
            # 如果是tensor，直接转换为numpy
            image_input = image.cpu().numpy().astype(np.float32)
        else:
            # 如果是PIL图像，使用preprocess
            image_input = self.preprocess(image).numpy().astype(np.float32)
        
        if self.is_dynamic:
            # 动态轴模式：直接处理整个batch
            output = self.visual_session.run(
                [self.visual_output_name],
                {self.visual_input_name: image_input}
            )[0]
            return F.normalize(torch.from_numpy(output).cuda(), dim=-1)
        else:
            # 固定轴模式：逐个处理
            results = []
            for i in range(image_input.shape[0]):
                output = self.visual_session.run(
                    [self.visual_output_name],
                    {self.visual_input_name: image_input[i:i+1]}
                )[0]
                results.append(output)
            return F.normalize(torch.from_numpy(np.concatenate(results, axis=0)).cuda(), dim=-1)
    
    @torch.no_grad()
    def encode_text(self, text):
        """根据text_path类型选择处理方式"""
        if isinstance(self.text_path, str):
            # 原有的ONNX处理逻辑
            if isinstance(text, torch.Tensor):
                text_input = text.cpu().numpy().astype(np.int32)
            else:
                if isinstance(text, str):
                    text = [text]
                text_tokens = self.tokenizer(text)
                text_input = text_tokens.numpy().astype(np.int32)
            
            if self.is_dynamic:
                output = self.text_session.run(
                    [self.text_output_name],
                    {self.text_input_name: text_input}
                )[0]
                text_features = torch.from_numpy(output).cuda()
            else:
                results = []
                for i in range(text_input.shape[0]):
                    output = self.text_session.run(
                        [self.text_output_name],
                        {self.text_input_name: text_input[i:i+1]}
                    )[0]
                    results.append(output)
                text_features = torch.from_numpy(np.concatenate(results, axis=0)).cuda()
        else:
            # 使用SplitTextEncoder处理
            text_features = self.text_encoder.encode_text(text)
            
        return F.normalize(text_features, dim=-1)

class SplitTextEncoder(nn.Module):
    """处理拆分后的文本编码器（base + adapter）"""
    def __init__(self, base_path, adapter_path, model_arch, batch_size=64):
        super().__init__()
        self.model_arch = model_arch
        self.batch_size = batch_size
        self.tokenizer = open_clip.get_tokenizer(model_arch)
        
        # ONNX Runtime配置
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }),
            'CPUExecutionProvider'
        ]
        
        # 初始化base和adapter会话
        self.base_session = ort.InferenceSession(
            base_path, 
            sess_options=sess_options,
            providers=providers
        )
        self.adapter_session = ort.InferenceSession(
            adapter_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # 获取输入输出名称
        self.base_input_name = self.base_session.get_inputs()[0].name
        self.base_output_name = self.base_session.get_outputs()[0].name
        self.adapter_input_name = self.adapter_session.get_inputs()[0].name
        self.adapter_output_name = self.adapter_session.get_outputs()[0].name
        
        # 检查是否是动态轴模型
        self.is_dynamic = 'dynamic' in base_path.lower()
        
        # 预热模型
        self._warmup()
    
    def _warmup(self):
        """预热模型以优化性能"""
        print("\n预热拆分文本编码器...")
        dummy_text = ["warm up text"] * self.batch_size
        self.encode_text(dummy_text)
        print("拆分文本编码器预热完成")
    
    @torch.no_grad()
    def encode_text(self, text):
        """串联方式处理文本编码"""
        # 处理输入类型
        if isinstance(text, torch.Tensor):
            text_input = text.cpu().numpy().astype(np.int32)
        else:
            if isinstance(text, str):
                text = [text]
            # 文本tokenization
            text_tokens = self.tokenizer(text)
            text_input = text_tokens.numpy().astype(np.int32)
        
        if self.is_dynamic:
            # 动态轴模式：直接处理整个batch
            # 1. Base encoder处理
            base_output = self.base_session.run(
                [self.base_output_name],
                {self.base_input_name: text_input}
            )[0]
            
            # 2. Adapter处理 - 使用base encoder的输出作为输入
            adapter_output = self.adapter_session.run(
                [self.adapter_output_name],
                {self.adapter_input_name: base_output}  # 直接使用base的输出
            )[0]
            
            # 3. 转换为tensor并归一化
            text_features = torch.from_numpy(adapter_output).cuda()
            return F.normalize(text_features, dim=-1)
        else:
            # 固定轴模式：逐个处理
            results = []
            for i in range(text_input.shape[0]):
                # Base encoder处理
                base_output = self.base_session.run(
                    [self.base_output_name],
                    {self.base_input_name: text_input[i:i+1]}
                )[0]
                
                # Adapter处理 - 使用base encoder的输出作为输入
                adapter_output = self.adapter_session.run(
                    [self.adapter_output_name],
                    {self.adapter_input_name: base_output}  # 直接使用base的输出
                )[0]
                
                results.append(adapter_output)
            
            # 合并结果并归一化
            text_features = torch.from_numpy(np.concatenate(results, axis=0)).cuda()
            return F.normalize(text_features, dim=-1)

def create_model(model_arch, model_path, resolution=256, batch_size=64):
    """创建模型实例"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. ONNX模型判断 - 保持原有的完整逻辑
    if isinstance(model_path, (list, tuple)) and any('onnx' in str(p).lower() for p in model_path):
        print("\n检测到ONNX模型...")
        
        # 检查是否有3个模型路径（visual + text_base + text_adapter）
        if len(model_path) == 3:
            # 首先找到visual路径
            visual_path = next(p for p in model_path if 'visual' in str(p).lower())
            text_paths = [p for p in model_path if p != visual_path]  # 剩余的两个路径
            
            # 然后从剩余路径中找到base和adapter
            base_path = next(p for p in text_paths if 'base' in str(p).lower())
            adapter_path = next(p for p in text_paths if p != base_path)  # 剩下的就是adapter路径
            
            print("\n检测到完整的模型集...")
            print(f"- Visual encoder路径: {visual_path}")
            print(f"- Text base encoder路径: {base_path}")
            print(f"- Text adapter路径: {adapter_path}")
            
            # 创建拆分文本编码器
            text_encoder = SplitTextEncoder(
                base_path=base_path,
                adapter_path=adapter_path,
                model_arch=model_arch,
                batch_size=batch_size
            )
            
            # 创建完整的ONNX包装器
            model = ONNXModelWrapper(
                visual_path=visual_path,
                text_path=text_encoder,  # 直接传入拆分文本编码器
                model_arch=model_arch,
                resolution=resolution,
                batch_size=batch_size
            )
            return model, model.preprocess, device
            
        # 检查是否是拆分的文本编码器（base + adapter）
        elif len(model_path) == 2 and all('text' in str(p).lower() for p in model_path):
            print("\n检测到拆分的文本编码器...")
            base_path = next(p for p in model_path if 'base' in p.lower())
            adapter_path = next(p for p in model_path if 'adapter' in p.lower())
            print(f"- Base encoder路径: {base_path}")
            print(f"- Adapter路径: {adapter_path}")
            
            # 创建拆分文本编码器
            model = SplitTextEncoder(
                base_path=base_path,
                adapter_path=adapter_path,
                model_arch=model_arch,
                batch_size=batch_size
            )
            return model, None, device
            
        # 处理其他ONNX模型情况
        if isinstance(model_path, (list, tuple)) and len(model_path) == 2:
            path1, path2 = model_path
            if 'text' in path1.lower() and 'visual' in path2.lower():
                text_path, visual_path = path1, path2
            elif 'visual' in path1.lower() and 'text' in path2.lower():
                visual_path, text_path = path1, path2
            else:
                raise ValueError("当提供两个路径时，文件名必须包含'text'和'visual'关键字")
        else:
            visual_path = model_path[0] if isinstance(model_path, (list, tuple)) else model_path
            text_path = visual_path.replace('_visual.onnx', '_text.onnx')
        
        print("\n使用ONNX Runtime进行推理...")
        print(f"- 视觉模型路径: {visual_path}")
        print(f"- 文本模型路径: {text_path}")
        
        model = ONNXModelWrapper(
            visual_path=visual_path,
            text_path=text_path,
            model_arch=model_arch,
            resolution=resolution,
            batch_size=batch_size
        )
        return model, model.preprocess, device
        
    else:
        # 2. 处理输入路径
        model_path_str = str(model_path[0] if isinstance(model_path, (list, tuple)) else model_path)
        
        # 3. 加载权重文件
        checkpoint = torch.load(model_path_str)
        
        # 4. 基于路径的模型类型判断
        if 'pruning' in model_path_str:
            print("\n检测到Pruning模型...")
            if 'px-ntk-pruning' in model_path_str:
                print("使用px-ntk-pruning方法...")
                # 创建基础模型
                model, _, transform = open_clip.create_model_and_transforms(
                    model_name=model_arch,
                    pretrained=None,
                    # image_mean=(0, 0, 0),
                    # image_std=(1, 1, 1),
                    # image_interpolation="bilinear",
                    force_image_size=(resolution, resolution)
                )
                
                # 检查是否需要重参数化
                if '_reparam' in model_path_str:
                    print("检测到_reparam后缀，对模型进行重参数化...")
                    model = reparameterize_model(model)
                
                # 加载权重
                try:
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 移除不需要的键
                    state_dict = {k: v for k, v in state_dict.items() 
                                if not k.endswith('.is_pruned') and k != '_metadata'}
                    
                    # 加载权重
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
                    if len(missing_keys) > 0:
                        print(f"警告: 缺失的键: {missing_keys}")
                    if len(unexpected_keys) > 0:
                        print(f"警告: 意外的键: {unexpected_keys}")
                        
                    print("px-ntk-pruning剪枝后的模型加载成功")
                    
                except Exception as e:
                    print(f"警告: px-ntk-pruning剪枝后的模型加载失败: {str(e)}")
                    raise e
            else:
                print("使用其他pruning方法...")
                # 创建基础模型
                model, _, transform = open_clip.create_model_and_transforms(
                    model_name=model_arch,
                    pretrained=None,  # 不加载预训练权重
                    # image_mean=(0, 0, 0),
                    # image_std=(1, 1, 1),
                    # image_interpolation="bilinear",
                    force_image_size=(resolution, resolution)
                )
                
                # 根据剪枝信息修改模型结构
                pruning_info = checkpoint['_metadata']['pruning_info']
                for name, info in pruning_info.items():
                    try:
                        # 使用eval()安全地获取模块
                        parts = name.split('.')
                        current = model
                        for part in parts[:-1]:
                            current = getattr(current, part)
                        
                        # 获取目标模块
                        target_module = getattr(current, parts[-1])
                        if hasattr(target_module, 'fc1'):
                            # 创建新的卷积层
                            new_fc1 = nn.Conv2d(
                                info['in_chs'],
                                info['target_hidden'],
                                kernel_size=1
                            )
                            new_fc2 = nn.Conv2d(
                                info['target_hidden'],
                                info['out_chs'],
                                kernel_size=1
                            )
                            
                            # 替换原有层
                            target_module.fc1 = new_fc1
                            target_module.fc2 = new_fc2
                            
                    except Exception as e:
                        print(f"警告: 修改模块 {name} 失败: {str(e)}")
                        continue
                
                # 加载权重
                try:
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 移除不需要的键
                    state_dict = {k: v for k, v in state_dict.items() 
                                if not k.endswith('.is_pruned') and k != '_metadata'}
                    
                    # 处理维度不匹配的权重
                    model_state_dict = model.state_dict()
                    for key in list(state_dict.keys()):
                        if key in model_state_dict:
                            # 如果权重为None，用零张量替换
                            if state_dict[key] is None:
                                state_dict[key] = torch.zeros_like(model_state_dict[key])
                                continue
                                
                            if state_dict[key].shape != model_state_dict[key].shape:
                                if 'conv.bn.weight' in key or 'conv.bn.bias' in key:
                                    # 处理bn权重维度不匹配
                                    orig_shape = state_dict[key].shape
                                    target_shape = model_state_dict[key].shape
                                    
                                    # 如果原始权重是多维的
                                    if len(orig_shape) > 1:
                                        # 先计算总元素数是否匹配
                                        orig_size = state_dict[key].numel()
                                        target_size = model_state_dict[key].numel()
                                        
                                        if orig_size == target_size:
                                            # 如果元素总数相同，直接reshape
                                            state_dict[key] = state_dict[key].reshape(target_shape)
                                        else:
                                            # 如果元素总数不同，需要进行均值池化
                                            if len(orig_shape) == 3:  # [C, H, W]
                                                # 对H和W维度进行平均池化
                                                pooled = state_dict[key].mean(dim=(1, 2))
                                                if pooled.shape == target_shape:
                                                    state_dict[key] = pooled
                                                else:
                                                    # 如果还是不匹配，创建新的权重
                                                    state_dict[key] = torch.ones_like(model_state_dict[key])
                                            else:
                                                # 其他情况，创建新的权重
                                                state_dict[key] = torch.ones_like(model_state_dict[key])
                                                
                                elif 'conv.bn.running_mean' in key or 'conv.bn.running_var' in key:
                                    # 处理running_mean和running_var
                                    if state_dict[key] is None:
                                        state_dict[key] = torch.zeros_like(model_state_dict[key])
                                    else:
                                        orig_size = state_dict[key].numel()
                                        target_size = model_state_dict[key].numel()
                                        
                                        if orig_size == target_size:
                                            state_dict[key] = state_dict[key].reshape(model_state_dict[key].shape)
                                        else:
                                            # 如果大小不匹配，使用默认值
                                            state_dict[key] = torch.zeros_like(model_state_dict[key]) if 'running_mean' in key else torch.ones_like(model_state_dict[key])
                    
                    # 加载处理后的权重
                    model.load_state_dict(state_dict)
                    print("其他pruning模型加载成功")
                except Exception as e:
                    print(f"警告: 其他pruning模型加载失败: {str(e)}")
                    raise e
                
        elif 'visionzip' in model_path_str:
            print("\n检测到VisionZip模型...")           
            
            if 'visionzip_siglip' in model_path_str:
                print("使用VisionZip SigLIP方法...")
                
                try:
                    from visionzip_siglip import modify_model
                    
                    # 从文件名提取参数
                    import re
                    print(f"正在从文件路径提取参数: {model_path_str}")
                    
                    # 提取dominant_num和contextual_num
                    d_match = re.search(r'd(\d+)', model_path_str)
                    c_match = re.search(r'c(\d+)', model_path_str)
                    b_match = re.search(r'b(\d+)', model_path_str)  # 提取block索引
                    
                    dominant_num = int(d_match.group(1)) if d_match else 64
                    contextual_num = int(c_match.group(1)) if c_match else 10
                    target_blocks = [int(b_match.group(1))] if b_match else None  # 默认使用最后一层
                    
                    print(f"Dominant tokens: {dominant_num}")
                    print(f"Contextual tokens: {contextual_num}")
                    print(f"Target blocks: {target_blocks}")
                    
                    # 1. 创建基础模型
                    model, _, transform = flair.create_model_and_transforms(
                        model_arch,
                        pretrained=None,
                        image_mean=(0.5, 0.5, 0.5), 
                        image_std=(0.5, 0.5, 0.5),   
                        image_interpolation="bicubic",
                        image_resize_mode="squash",
                        force_image_size=(resolution, resolution),
                    )
                    
                    # 2. 先修改模型结构
                    print("\n修改模型结构为VisionZip...")
                    model = modify_model(
                        model, 
                        dominant_num=dominant_num, 
                        contextual_num=contextual_num,
                        target_blocks=target_blocks
                    )
                    
                    # 3. 然后加载VisionZip权重
                    print("\n加载VisionZip权重...")
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    print(f"\nVisionZip SigLIP模型创建完成:")
                    print(f"- Dominant tokens: {dominant_num}")
                    print(f"- Contextual tokens: {contextual_num}")
                    print(f"- Target blocks: {target_blocks}")
                    
                except Exception as e:
                    print(f"警告: VisionZip SigLIP修改失败: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    raise e
                
            elif 'visionzip_vanilla' in model_path_str:
                print("使用VisionZip Vanilla方法...")

                try:
                    from custom_timm.models import fastvit
                    print("预加载FastViT模块成功")
                except Exception as e:
                    print(f"警告: 预加载FastViT模块失败: {str(e)}")
                    
                model, _, transform = flair.create_model_and_transforms(
                    model_arch,
                    pretrained=None,  # 不加载预训练权重
                    image_mean=(0, 0, 0),
                    image_std=(1, 1, 1),
                    image_interpolation="bilinear",
                    force_image_size=(resolution, resolution)
                )

                try:
                    from visionzip_vanilla.fastvit_visionzip import convert_fastvit_to_visionzip
                    # 从文件名提取参数
                    import re
                    print(f"检测到Vanilla VisionZip模型，正在从文件路径提取参数: {model_path_str}")

                    # 提取dominant_num和contextual_num
                    dominant_match = re.search(r'dominant(\d+)', model_path_str)
                    contextual_match = re.search(r'contextual(\d+)', model_path_str)
                    
                    dominant_num = int(dominant_match.group(1)) if dominant_match else 8
                    contextual_num = int(contextual_match.group(1)) if contextual_match else 8
                    
                    print(f"Dominant tokens: {dominant_num}")
                    print(f"Contextual tokens: {contextual_num}")
                    
                    # 转换模型
                    model.visual = convert_fastvit_to_visionzip(
                        model.visual,
                        enable_compression=True,
                        dominant_num=dominant_num,
                        contextual_num=contextual_num
                    )
                    print(f"\n使用以下参数应用Vanilla VisionZip修改:")
                    print(f"- Dominant tokens: {dominant_num}")
                    print(f"- Contextual tokens: {contextual_num}")
                    
                except Exception as e:
                    print(f"警告: Vanilla VisionZip修改失败: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    
            elif 'visionzip_mobileclip' in model_path_str:
                print("使用VisionZip MobileCLIP方法...")

                model, _, transform = flair.create_model_and_transforms(
                    model_arch,
                    pretrained=None,  # 不加载预训练权重
                    image_mean=(0, 0, 0),
                    image_std=(1, 1, 1),
                    image_interpolation="bilinear",
                    force_image_size=(resolution, resolution)
                )

                try:
                    # 检查是否是新版本的VisionZip模型
                    if "visionzip_stages" in model_path_str:
                        print(f"检测到visionzip_mobileclip方法")
                        from visionzip_mobileclip.main import visionzip_onnx_mobileclip
                        from visionzip_mobileclip.utils import StageCompressionConfig   
                        # 加载配置文件
                        config_path = os.path.join(os.path.dirname(model_path_str), "visionzip_config.json")
                        if os.path.exists(config_path):
                            print(f"找到配置文件: {config_path}")
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            compression_config = StageCompressionConfig(config['stage_configs'])
                            model = visionzip_onnx_mobileclip(model, compression_config=compression_config)
                            print(f"\n使用配置文件中的参数应用Mobile VisionZip修改:")
                            print(f"启用的stages: {config['enabled_stages']}")
                        else:
                            print(f"警告: 未找到配置文件 {config_path}")
                            # 尝试在outputs/visionzip_mobileclip目录下查找
                            alt_config_path = os.path.join("outputs", "visionzip_mobileclip", "visionzip_config.json")
                            if os.path.exists(alt_config_path):
                                print(f"在outputs/visionzip_mobileclip目录下找到配置文件")
                                with open(alt_config_path, 'r') as f:
                                    config = json.load(f)
                                compression_config = StageCompressionConfig(config['stage_configs'])
                                model = visionzip_onnx_mobileclip(model, compression_config=compression_config)
                                print(f"\n使用配置文件中的参数应用Mobile VisionZip修改:")
                                print(f"启用的stages: {config['enabled_stages']}")
                            else:
                                print(f"警告: 在outputs/visionzip_mobileclip目录下也未找到配置文件")
                    else:
                        print(f"检测到visionzip_mobileclip_stage3方法")
                        # 旧版本的参数提取方式
                        from visionzip_mobileclip_stage3.main import visionzip_onnx_mobileclip
                        import re
                        # 提取temperature
                        temp_match = re.search(r'temp(0\.\d+)', model_path_str)
                        temperature = float(temp_match.group(1)) if temp_match else 0.01
                        print(f"Temperature: {temperature}")
                        
                        # 提取ratio
                        ratio_match = re.search(r'ratio(0\.\d+)', model_path_str)
                        keep_ratio = float(ratio_match.group(1)) if ratio_match else 0.7
                        use_keep_ratio = True if ratio_match else False
                        print(f"Keep ratio: {keep_ratio}, Use keep ratio: {use_keep_ratio}")
                        
                        # 提取spatial_weight
                        spatial_match = re.search(r'spatial(0\.\d+)', model_path_str)
                        spatial_weight = float(spatial_match.group(1)) if spatial_match else 0.3
                        print(f"Spatial weight: {spatial_weight}")
                        
                        model = visionzip_onnx_mobileclip(
                            model, 
                            temperature=temperature,
                            use_keep_ratio=use_keep_ratio,
                            keep_ratio=keep_ratio,
                            spatial_weight=spatial_weight
                        )
                        print(f"\n使用以下参数应用Mobile VisionZip修改:")
                        print(f"- Temperature: {temperature}")
                        print(f"- Keep ratio: {keep_ratio} (enabled: {use_keep_ratio})")
                        print(f"- Spatial weight: {spatial_weight}")
                        
                except Exception as e:
                    print(f"警告: Mobile VisionZip修改失败: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
            
            # 加载权重
            try:
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("VisionZip模型加载成功")
            
            except Exception as e:
                print(f"警告: VisionZip模型加载失败: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                raise e
                
        # 5. 原始模型判断
        elif 'checkpoints' in model_path_str:
            if 'siglip' in str(model_arch).lower():
                print("\n加载SigLIP原始模型...")
                model_path = model_path[0] if isinstance(model_path, (list, tuple)) else model_path
                print(f"模型路径: {model_path}")
                
                # SigLIP特殊配置
                model, _, transform = flair.create_model_and_transforms(
                    model_arch,
                    pretrained=model_path,
                    image_mean=(0.5, 0.5, 0.5), 
                    image_std=(0.5, 0.5, 0.5),   
                    image_interpolation="bicubic",
                    image_resize_mode="squash",
                    force_image_size=(resolution, resolution),
                )
                print(f"模型结构: {model}")
                
            elif 'vit_b_32_256' in model_path_str:
                print("\n加载openclip的原始ViT模型...")
                model_path = model_path[0] if isinstance(model_path, (list, tuple)) else model_path
                print(f"模型路径: {model_path}")
                model, _, transform = flair.create_model_and_transforms(
                    model_arch,
                    pretrained=model_path,
                    force_image_size=(resolution, resolution)
                )
            else:
                print("\n加载原始MobileCLIP模型...")
                model_path = model_path[0] if isinstance(model_path, (list, tuple)) else model_path
                print(f"模型路径: {model_path}")
                model, _, transform = flair.create_model_and_transforms(
                    model_arch,
                    pretrained=model_path,
                    image_mean=(0, 0, 0),
                    image_std=(1, 1, 1),
                    image_interpolation="bilinear",
                    force_image_size=(resolution, resolution)
                )
                model = reparameterize_model(model)
        
        else:
            raise ValueError(f"无法识别的模型类型: {model_path_str}")

        # 6. 返回处理后的模型
        model.eval()
        model = model.to(device)
        return model, transform, device

def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        task="zeroshot_retrieval",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader

def evaluate_classification_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    resolution=256,
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = create_model(
        model_arch=model_arch, 
        model_path=model_path,
        resolution=resolution,
        batch_size=batch_size
    )

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
            zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    metrics = zsc.evaluate(
        model,
        dataloader,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
        amp=False
    )
    metrics['mean_per_class_recall'] = float(metrics['mean_per_class_recall'])

    return metrics

def evaluate_retrieval_webdataset(
    task, 
    model_arch, 
    model_path, 
    data_root=None, 
    dataset_len=None, 
    batch_size=64,
    resolution=256
):
    """Evaluate CLIP model on retrieval task."""
    print(f"\n=== 开始评估 {task} ===")
    print(f"- 模型架构: {model_arch}")
    print(f"- 模型路径: {model_path}")
    print(f"- 批处理大小: {batch_size}")
    print(f"- 输入分辨率: {resolution}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with parameters
    model, transform, device = create_model(
        model_arch=model_arch, 
        model_path=model_path,
        resolution=resolution,
        batch_size=batch_size
    )

    # Load data
    print("\n加载数据集...")
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, 
        batch_size=batch_size,
    )

    # Evaluate
    print("\n开始评估...")
    metrics = zsr.evaluate(
        model,
        dataloader,
        open_clip.get_tokenizer(model_arch),
        recall_k_list=[1, 5, 10, 20],
        device=device,
        amp=False
    )
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webdataset evaluation script.")
    parser = parse_args(parser)
    args = parser.parse_args()

    for task in args.eval_tasks:
        if task == "imagenet1k":
            metric = evaluate_classification_webdataset(
                task=task, 
                model_arch=args.model_arch, 
                model_path=args.model_path,
                data_root=args.data_root, 
                batch_size=args.batch_size,
                resolution=args.resolution
            )
            print(f"ImageNet1K Eval Metrics: {metric}")
        else:
            metric = evaluate_retrieval_webdataset(
                task=task, 
                model_arch=args.model_arch, 
                model_path=args.model_path,
                data_root=args.data_root, 
                batch_size=args.batch_size,
                resolution=args.resolution
            )
            print(f"{task.capitalize()} Eval Metrics: {metric}")