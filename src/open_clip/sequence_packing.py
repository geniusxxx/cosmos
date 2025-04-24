"""
序列打包实现，提供对多序列的高效处理。
基于xFormers的BlockDiagonalMask实现。
"""

import torch
import warnings
from typing import List, Tuple, Dict, Optional, Union, Any

try:
    from xformers.ops import fmha
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    warnings.warn(
        "xFormers未安装，序列打包功能将被禁用。安装xFormers可加速训练。"
        "可使用: pip install xformers>=0.0.20"
    )


def pack_sequences(sequence_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
    """打包多个token序列为单个张量，并创建块对角掩码。
    
    Args:
        sequence_list: 要打包的token序列列表，每个序列形状为[B, S_i, D]
        
    Returns:
        Tuple包含:
            - packed_tensor: 所有序列打包后的张量 [Sum(B_i), S_i, D]
            - block_mask: xFormers的块对角掩码
    """
    if not HAS_XFORMERS:
        raise ImportError("xFormers未安装，无法使用序列打包功能")
        
    # 创建块对角掩码并打包输入
    block_mask, packed_tensor = fmha.BlockDiagonalMask.from_tensor_list(sequence_list)
    return packed_tensor, block_mask


def unpack_sequences(packed_output: torch.Tensor, block_mask: Any) -> List[torch.Tensor]:
    """将打包的输出张量解包回原始序列列表。
    
    Args:
        packed_output: 模型处理打包序列的输出
        block_mask: 打包时使用的块对角掩码
        
    Returns:
        解包后的序列张量列表
    """
    if not HAS_XFORMERS:
        raise ImportError("xFormers未安装，无法解包序列")
        
    return block_mask.split(packed_output)


class SequencePackingWrapper:
    """序列打包包装器，为模块提供序列打包能力"""
    
    def __init__(self, module, use_packing=True):
        """初始化序列打包包装器
        
        Args:
            module: 要包装的模块
            use_packing: 是否启用序列打包
        """
        self.module = module
        self.use_packing = use_packing and HAS_XFORMERS
    
    def __call__(self, sequence_list):
        """处理序列列表
        
        Args:
            sequence_list: token序列列表
            
        Returns:
            处理后的序列列表
        """
        if not self.use_packing or len(sequence_list) <= 1:
            # 单个序列无需打包
            return [self.module(seq) for seq in sequence_list]
        
        try:
            # 打包序列
            packed_tensor, block_mask = pack_sequences(sequence_list)
            
            # 通过模块前向传播
            packed_output = self.module(packed_tensor)
            
            # 解包输出
            outputs = unpack_sequences(packed_output, block_mask)
            
            return outputs
        except Exception as e:
            warnings.warn(f"序列打包失败: {e}，退化为顺序处理")
            # 出错时退化为顺序处理
            return [self.module(seq) for seq in sequence_list]


def group_by_size(input_list):
    """按尺寸分组图像列表
    
    Args:
        input_list: 输入张量列表
        
    Returns:
        按照最后一维尺寸分组的列表
    """
    if not input_list:
        return []
    
    size_groups = {}
    for item in input_list:
        size = item.shape[-1] if hasattr(item, 'shape') else None
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(item)
        
    return list(size_groups.values()) 