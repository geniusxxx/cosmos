import torch
import sys
import copy
import torch.nn as nn
import onnx
import argparse
import open_clip
from src.open_clip import create_model_and_transforms, get_tokenizer

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

class VisualEncoder:
    def __init__(self, model, preprocess, reparam=True, model_arch=None, normalize=True, framework='openclip'):
        self.reparam = reparam
        self.model_arch = model_arch
        self.normalize = normalize  # 添加normalize参数
        self.framework = framework  # 添加framework参数
        self.encoder = self._get_visual_encoder(model)
        self.encoder.eval()
        self.output_path = None
        self.preprocess = preprocess  # 保存预处理函数
    
    def _get_visual_encoder(self, model):
        visual_encoder = model.visual
        if self.framework == 'mobileclip':
            if self.reparam:
                if self.model_arch and 'repvit' in self.model_arch.lower():
                    print("检测到RepVit模型，正在进行融合...")
                    visual_encoder = self._fuse_model(visual_encoder)
                    print("模型融合完成.")
                    print("\n模型参数:")
                    for name, module in visual_encoder.named_modules():
                        if isinstance(module, nn.Conv2d):
                            print(f"{name}: in={module.in_channels}, out={module.out_channels}, "
                                f"kernel={module.kernel_size}, stride={module.stride}")
                else:
                    print("检测到FastVit模型，正在进行重参数化...")
                    visual_encoder = self._reparameterize_model(visual_encoder)
                    print("模型重参数化完成.")
                    print(f"visual_encoder: {visual_encoder}")

        # 包装带手动L2归一化的编码器
        class NormalizedEncoder(nn.Module):
            def __init__(self, base_encoder, normalize):
                super().__init__()
                self.base_encoder = base_encoder
                self.normalize = normalize

            def forward(self, x):
                features = self.base_encoder(x)
     # 处理不同类型的输出
                if isinstance(features, tuple):
                    # 元组格式 - 典型的格式是(tokens, features)
                    # 根据COSMOS模型，我们需要第二个元素
                    features = features[1]  # 可能是features[0]，具体看模型实现
                elif isinstance(features, dict):
                    # 字典格式 - 提取image_features
                    features = features['image_features']
                if self.normalize:
                    # 手动实现L2归一化
                    square = features * features  # Mul 操作
                    sum_square = torch.sum(square, dim=-1, keepdim=True)  # ReduceSum 操作
                    sqrt = torch.sqrt(sum_square)  # Sqrt 操作
                    features = features / sqrt  # Div 操作
                return features

        return NormalizedEncoder(visual_encoder, self.normalize)
    
    def verify_outputs(self, test_image):
        """验证PyTorch和ONNX输出是否一致"""
        import numpy as np
        import onnxruntime
        
        # PyTorch预处理和推理
        self.encoder.eval()
        with torch.no_grad():
            # 使用相同的预处理
            processed_input = self.preprocess(test_image).unsqueeze(0)
            pytorch_output = self.encoder(processed_input)
        
        # ONNX预处理和推理
        ort_session = onnxruntime.InferenceSession(self.output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: processed_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较输出
        pytorch_output = pytorch_output.cpu().numpy()
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        print(f"\nOutput Verification Results:")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # 打印更多信息以帮助调试
        print("\nInput tensor info:")
        print(f"Shape: {processed_input.shape}")
        print(f"Range: [{processed_input.min():.3f}, {processed_input.max():.3f}]")
        print(f"Mean: {processed_input.mean():.3f}")
        print(f"Std: {processed_input.std():.3f}")
        
        print("\nOutput tensor info:")
        print(f"PyTorch shape: {pytorch_output.shape}")
        print(f"ONNX shape: {onnx_output.shape}")
        print(f"PyTorch range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
        print(f"ONNX range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
        
        return max_diff < 1e-5
    
    def export_onnx(self, output_path, resolution=256, verbose=False, verify=False, test_image=None, dynamic_axes=False):
        # print(f"\nVisual Encoder: {self.encoder}")
        
        dummy_input = torch.randn(1, 3, resolution, resolution)
        
        def get_shape_hook(name):
            def hook(model, input, output):
                # 只打印主要层的输出
                if name in ['trunk.stem.0', 'trunk.stem.1', 'trunk.stem.2', 
                           'trunk.stages.0', 'trunk.stages.1', 'trunk.stages.2', 
                           'trunk.stages.3', 'trunk.final_conv', 'head']:
                    print(f"{name} output shape: {output.shape}")
            return hook
        
        hook_handles = []
        def register_hooks(model, prefix=''):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                handle = module.register_forward_hook(get_shape_hook(full_name))
                hook_handles.append(handle)
                register_hooks(module, full_name)
        
        print("\n=== Visual Encoder Feature Shapes ===")
        print(f"dummy_input shape: {dummy_input.shape}\n")
        
        register_hooks(self.encoder)
        
        print("Running forward pass to get feature shapes...")
        with torch.no_grad():
            _ = self.encoder(dummy_input)
        print("=== End of Visual Encoder Feature Shapes ===\n")
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
            
        # 保存输出路径
        if '_visual' not in output_path:
            self.output_path = output_path.replace('.onnx', '_visual.onnx')
        else:
            self.output_path = output_path
            
        # 导出ONNX
        export_args = {
            'model': self.encoder,
            'args': dummy_input,
            'f': self.output_path,
            'opset_version': 18,
            'verbose': verbose,
            'export_params': True,
            'do_constant_folding': False,
            'input_names': ['input'],
            'output_names': ['image_features'],
        }
        
        if dynamic_axes:
            export_args['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'image_features': {0: 'batch_size'}
            }
            
        try:
            print(f"正在导出ONNX模型到 {self.output_path}...")
            torch.onnx.export(**export_args)
            print(f"导出完成!")
        except Exception as e:
            print(f"导出失败: {str(e)}")
            return False
        
        # 使用onnxsim简化模型
        try:
            print("\n使用onnxsim简化模型...")
            import onnxsim
            onnx_model = onnx.load(self.output_path)
            
            # 简化模型
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                print("模型简化成功，保存简化后的模型...")
                onnx.save(model_simp, self.output_path)
            else:
                print("警告: 模型简化失败，将使用原始模型")
                onnx.save(onnx_model, self.output_path)
        except Exception as e:
            print(f"警告: 模型简化过程中出错: {str(e)}")
            print("将使用原始模型")
            try:
                onnx.save(onnx_model, self.output_path)
            except:
                print("保存原始模型也失败，可能需要检查ONNX模型格式")
        
        # 验证ONNX模型
        try:
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model, full_check=True)
            print("Visual encoder ONNX model is valid.")
            
            # 仅在指定验证时执行
            if verify:
                print("\nVerifying outputs...")
                verification_result = self.verify_outputs(test_image)
                if verification_result:
                    print("Output verification passed! PyTorch and ONNX outputs are consistent.")
                else:
                    print("Warning: Output verification failed! PyTorch and ONNX outputs have significant differences.")
                return verification_result
            return True
        except onnx.checker.ValidationError as e:
            print("Visual encoder ONNX model is invalid.")
            print(f"Error: {e}")
            return False

    @staticmethod
    def _reparameterize_model(model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
        
    @staticmethod
    def _fuse_model(model: nn.Module) -> nn.Module:
        """reparameterize model for repvit."""
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "fuse"):
                module.fuse()
        return model

class TextEncoder:
    def __init__(self, model, tokenizer, normalize=True, framework='openclip'):
        self.normalize = normalize  # 添加normalize参数
        self.framework = framework  # 添加framework参数
        self.encoder = self._get_text_encoder(model)
        self.encoder.eval()
        self.output_path = None
        self.tokenizer = tokenizer  # 保存tokenizer
    
    def _get_text_encoder(self, model):
        # 根据框架类型选择文本编码器的获取方式
        if self.framework == 'mobileclip':
            text_encoder = model.text
            
            class MobileClipTextEncoder(nn.Module):
                def __init__(self, encoder, normalize):
                    super().__init__()
                    self.encoder = encoder
                    self.normalize = normalize
                    
                def forward(self, text):
                    features, _= self.encoder(text)
                    if self.normalize:
                        # 手动实现L2归一化
                        square = features * features  # Mul 操作
                        sum_square = torch.sum(square, dim=-1, keepdim=True)  # ReduceSum 操作
                        sqrt = torch.sqrt(sum_square)  # Sqrt 操作
                        features = features / sqrt  # Div 操作
                    return features
            
            return MobileClipTextEncoder(text_encoder, self.normalize)
        
        else:
            # 创建一个完整的文本编码器，包含所有必要的组件
            class CustomTextEncoder(nn.Module):
                def __init__(self, clip_model, normalize):
                    super().__init__()
                    # 提取CLIP模型中的文本相关组件
                    self.token_embedding = clip_model.token_embedding
                    self.positional_embedding = clip_model.positional_embedding
                    self.transformer = clip_model.transformer
                    self.ln_final = clip_model.ln_final
                    self.text_projection = clip_model.text_projection
                    self.register_buffer('attn_mask', clip_model.attn_mask)
                    self.text_pool_type = getattr(clip_model, 'text_pool_type', 'argmax')
                    self.normalize = normalize
                    # 检查是否需要返回token级特征
                    self.output_all = getattr(clip_model, 'output_all', False)
                    self.text_token_mapping = getattr(clip_model, 'text_token_mapping', None)

                def forward(self, text):
                    # 获取数据类型
                    cast_dtype = self.transformer.get_cast_dtype()
                    
                    # 确保输入是正确的整型
                    if text.dtype != torch.int32 and text.dtype != torch.int64:
                        text = text.to(torch.int32)
                    
                    # 1. 词嵌入
                    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
                    
                    # 2. 位置编码
                    x = x + self.positional_embedding.to(cast_dtype)
                    
                    # 3. Transformer处理
                    x = self.transformer(x, attn_mask=self.attn_mask)
                    
                    # 4. 最终层归一化
                    x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                    
                    # 5. 全局池化
                    from src.open_clip.transformer import text_global_pool
                    pooled, tokens = text_global_pool(x, text, self.text_pool_type)
                    
                    # 6. 应用投影（如果有）
                    if self.text_projection is not None:
                        if isinstance(self.text_projection, nn.Linear):
                            pooled = self.text_projection(pooled)
                        else:
                            pooled = pooled @ self.text_projection
                    
                    # 7. 应用归一化（如果需要）
                    if self.normalize:
                        # 手动实现L2归一化
                        square = pooled * pooled  # Mul 操作
                        sum_square = torch.sum(square, dim=-1, keepdim=True)  # ReduceSum 操作
                        sqrt = torch.sqrt(sum_square)  # Sqrt 操作
                        pooled = pooled / sqrt  # Div 操作
                    
                    return pooled

            # 返回自定义文本编码器
            return CustomTextEncoder(model, self.normalize)
    
    def verify_outputs(self, test_texts):
        """验证PyTorch和ONNX输出是否一致"""
        import numpy as np
        import onnxruntime
        
        # 逐个处理文本，确保batch size为1
        for single_text in test_texts:
            # 1. Tokenize单个文本
            text_tokens = self.tokenizer([single_text])
            
            # 确保tokens是int32类型，很多ONNX运行时需要这个
            if text_tokens.dtype != torch.int32:
                text_tokens = text_tokens.to(torch.int32)
            
            # 2. PyTorch推理
            with torch.no_grad():
                pytorch_output = self.encoder(text_tokens)
            
            # 3. ONNX推理
            ort_session = onnxruntime.InferenceSession(self.output_path)
            ort_inputs = {
                ort_session.get_inputs()[0].name: text_tokens.cpu().numpy()
            }
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # 4. 比较输出
            pytorch_output = pytorch_output.cpu().numpy()
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            print(f"\nOutput Verification Results for text: {single_text}")
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            # 5. 打印更多信息以帮助调试
            print(f"\nInput token info:")
            print(f"Shape: {text_tokens.shape}")
            print(f"Data type: {text_tokens.dtype}")
            
            print(f"\nOutput tensor info:")
            print(f"PyTorch shape: {pytorch_output.shape}")
            print(f"ONNX shape: {onnx_output.shape}")
            print(f"PyTorch range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
            print(f"ONNX range: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
            
            if max_diff >= 1e-5:
                return False
        
        return True
    
    def export_onnx(self, output_path, verbose=False, verify=False, test_texts=None, dynamic_axes=False):
        print("\n开始导出文本编码器...")
        
        # 使用正确的数据类型创建dummy input
        dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32)
        
        def get_shape_hook(name):
            def hook(model, input, output):
                # 处理tuple类型的输出
                if isinstance(output, tuple):
                    shapes = [o.shape if hasattr(o, 'shape') else type(o) for o in output]
                    print(f"{name} output shapes: {shapes}")
                else:
                    print(f"{name} output shape: {output.shape}")
            return hook
        
        hook_handles = []
        def register_hooks(model, prefix=''):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                handle = module.register_forward_hook(get_shape_hook(full_name))
                hook_handles.append(handle)
                register_hooks(module, full_name)
        
        print("\n=== 文本编码器特征形状 ===")
        print(f"输入形状: {dummy_input.shape}, 类型: {dummy_input.dtype}\n")
        
        register_hooks(self.encoder)
        
        print("运行前向传播获取特征形状...")
        with torch.no_grad():
            out = self.encoder(dummy_input)
            print(f"编码器输出形状: {out.shape}")
        print("=== 文本编码器特征形状结束 ===\n")
        
        # 移除钩子
        for handle in hook_handles:
            handle.remove()
            
        # 保存输出路径
        if '_text' not in output_path:
            self.output_path = output_path.replace('.onnx', '_text.onnx')
        else:
            self.output_path = output_path
            
        # Export to ONNX
        export_args = {
            'model': self.encoder,
            'args': dummy_input,
            'f': self.output_path,
            'opset_version': 18,
            'verbose': verbose,
            'export_params': True,
            'do_constant_folding': False,
            'input_names': ['input'],
            'output_names': ['text_features'],
        }
        
        if dynamic_axes:
            export_args['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'text_features': {0: 'batch_size'}
            }
            
        try:
            print(f"正在导出ONNX模型到 {self.output_path}...")
            torch.onnx.export(**export_args)
            print(f"导出完成!")
        except Exception as e:
            print(f"导出失败: {str(e)}")
            return False
        
        # 使用onnxsim简化模型
        print("\n使用onnxsim简化模型...")
        import onnxsim
        onnx_model = onnx.load(self.output_path)
        try:
            # 简化模型
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                print("模型简化成功，保存简化后的模型...")
                onnx.save(model_simp, self.output_path)
            else:
                print("警告: 模型简化失败，将使用原始模型")
                onnx.save(onnx_model, self.output_path)
        except Exception as e:
            print(f"警告: 模型简化过程中出错: {str(e)}")
            print("将使用原始模型")
            onnx.save(onnx_model, self.output_path)
        
        # 验证ONNX模型
        onnx_model = onnx.load(self.output_path)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
            print("文本编码器ONNX模型验证成功。")
            
            # 仅在指定验证时执行
            if verify:
                print("\n验证模型输出...")
                if test_texts is None:
                    test_texts = ["a photo of a cat", "this is a test"]
                verification_result = self.verify_outputs(test_texts)
                if verification_result:
                    print("输出验证通过! PyTorch和ONNX输出一致。")
                else:
                    print("警告: 输出验证失败! PyTorch和ONNX输出存在显著差异。")
                return verification_result
            return True
            
        except onnx.checker.ValidationError as e:
            print("文本编码器ONNX模型无效。")
            print(f"错误: {e}")
            return False

def parsers(args):
    parser = argparse.ArgumentParser(description='Export CLIP model to ONNX')
    parser.add_argument('--framework', type=str, default='openclip',
                      choices=['openclip', 'mobileclip'],
                      help='Framework to use: openclip or mobileclip')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-arch', type=str, default=None)
    parser.add_argument('--verbose-onnx', action='store_true')
    parser.add_argument('--output-path', type=str, default="./")
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--reparam', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=True)
    parser.add_argument('--export-image', action='store_true')
    parser.add_argument('--export-text', action='store_true')
    parser.add_argument('--export-all', action='store_true')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX output against PyTorch output')
    parser.add_argument('--dynamic-axes', action='store_true',
                       help='Export ONNX with dynamic axes for batch dimension')
    parser.add_argument('--normalize', type=lambda x: (str(x).lower() == 'true'),
                       choices=[True, False], default=True,
                       help='Whether to add normalization in the exported model')
    parser.add_argument('--model-type', type=str, choices=['teacher', 'student', 'auto'], default='auto',
                       help='指定导出teacher或student模型，auto表示优先使用teacher')
    return parser.parse_args(args)

def main(args):
    args = parsers(args)
    
    # 只在需要验证时才准备测试数据
    test_image = None
    test_texts = None
    if args.verify:
        from PIL import Image
        test_image = Image.open("/home/xuboyu/Projects/cosmos/assets/framework.png").convert('RGB')
        test_texts = ["a diagram", "a dog", "a cat"]

    # Load model and transforms
    model, _, preprocess_val = create_model_and_transforms(
        model_name=args.model_arch,
        pretrained=args.model_path,
        # image_mean=(0, 0, 0),
        # image_std=(1, 1, 1),
        # image_interpolation="bilinear",
        force_image_size=(args.resolution, args.resolution),
        output_all=True,
        cosmos=True,
        attentional_pool=True,
    )
    
    # 检查是否包含teacher模型权重，并优先使用teacher
    if args.model_path:
        if args.model_type == 'teacher':
            print("\n尝试加载teacher模型...")
            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'teacher' in checkpoint:
                print("找到teacher模型，加载teacher权重")
                sd_teacher = checkpoint["teacher"]
                if list(sd_teacher.keys())[0].startswith('module.'):
                    sd_teacher = {k[len('module.'):]: v for k, v in sd_teacher.items()}
                model.load_state_dict(sd_teacher)
                print("已成功加载teacher模型权重")
            elif isinstance(checkpoint, dict) and 'student' in checkpoint:
                print("未找到teacher模型，但找到student模型，当前使用student权重")
            else:
                print("未找到teacher或student模型权重，使用默认加载的权重")
        elif args.model_type == 'student':
            print("\n尝试加载student模型...")
            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'student' in checkpoint:
                print("找到student模型，加载student权重")
                sd_student = checkpoint["student"]
                if list(sd_student.keys())[0].startswith('module.'):
                    sd_student = {k[len('module.'):]: v for k, v in sd_student.items()}
                model.load_state_dict(sd_student)
                print("已成功加载student模型权重")
            elif isinstance(checkpoint, dict) and 'teacher' in checkpoint:
                print("未找到student模型，但找到teacher模型，当前使用teacher权重")
            else:
                print("未找到teacher或student模型权重，使用默认加载的权重")
        else:
            print(f"\n尝试加载模型权重时出错")
    
    # 确保模型处于评估模式
    model.eval()
    print(f"model: {model}")

    export_results = []
    
    # Export visual encoder
    if args.export_all or args.export_image:
        print("\nExporting visual encoder...")
        visual_encoder = VisualEncoder(
            model=model,
            preprocess=preprocess_val, 
            reparam=args.reparam, 
            model_arch=args.model_arch,
            normalize=args.normalize,
            framework=args.framework
        )
        visual_result = visual_encoder.export_onnx(
            output_path=args.output_path,
            resolution=args.resolution,
            verbose=args.verbose_onnx,
            verify=args.verify,
            test_image=test_image if args.verify else None,
            dynamic_axes=args.dynamic_axes
        )
        export_results.append(('Visual Encoder', visual_result))
    
    # Export text encoder
    if args.export_all or args.export_text:
        print("\nExporting text encoder...")
        tokenizer = get_tokenizer(args.model_arch)
        text_encoder = TextEncoder(
            model=model,
            tokenizer=tokenizer,
            normalize=args.normalize,
            framework=args.framework
        )
        text_result = text_encoder.export_onnx(
            output_path=args.output_path,
            verbose=args.verbose_onnx,
            verify=args.verify,
            test_texts=test_texts if args.verify else None,
            dynamic_axes=args.dynamic_axes
        )
        export_results.append(('Text Encoder', text_result))
    
    # Print summary
    print("\nExport Summary:")
    for name, success in export_results:
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main(sys.argv[1:])