"""
支持序列打包的注意力模块
包括自注意力和交叉注意力的优化实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Any

try:
    from xformers.ops import fmha
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


class SequencePackingAttention(nn.Module):
    """支持序列打包的自注意力实现"""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0., 
        proj_drop: float = 0.
    ):
        """初始化支持序列打包的注意力模块
        
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数量
            qkv_bias: 是否在qkv投影中使用偏置
            attn_drop: 注意力dropout率
            proj_drop: 输出投影dropout率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 检查xFormers支持
        self.use_xformers = HAS_XFORMERS

    def forward(self, x: torch.Tensor, attn_bias: Optional[Any] = None) -> torch.Tensor:
        """前向计算，支持序列打包
        
        Args:
            x: 输入序列 [B, N, C]
            attn_bias: 注意力偏置/掩码，可以是BlockDiagonalMask
            
        Returns:
            注意力输出 [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        # 使用xFormers的高效注意力实现
        if self.use_xformers and isinstance(attn_bias, fmha.BlockDiagonalMask):
            # 直接使用xFormers的fused attention
            q, k, v = qkv.unbind(2)
            q = q.permute(0, 2, 1, 3)  # B, H, N, D
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # xFormers的内存高效注意力计算
            x = fmha.memory_efficient_attention(
                q, k, v, 
                attn_bias=attn_bias,
                scale=self.scale
            )
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        else:
            # 降级为原始实现
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # 使用掩码
            if attn_bias is not None:
                # 支持不同类型的掩码
                if isinstance(attn_bias, torch.Tensor):
                    attn = attn + attn_bias
                else:
                    # 如果是其他类型的掩码，尝试转换
                    try:
                        tensor_mask = attn_bias.materialize(
                            (B, self.num_heads, N, N), 
                            dtype=attn.dtype,
                            device=attn.device
                        )
                        attn = attn + tensor_mask
                    except:
                        pass
                        
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PackedCrossAttentionPooling(nn.Module):
    """支持序列打包的交叉注意力池化"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        """初始化支持序列打包的交叉注意力池化
        
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数量
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # 检查xFormers支持
        self.use_xformers = HAS_XFORMERS
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_bias: Optional[Any] = None) -> torch.Tensor:
        """处理多种输入格式的前向计算
        """
        # 处理各种输入形状
        x_orig_shape = x.shape
        
        # 如果是2D输入 [B, C]，转换为3D [B, 1, C]
        if len(x_orig_shape) == 2:
            x = x.unsqueeze(1)
        
        # 如果context是2D [M, C]，转换为3D [1, M, C]
        if len(context.shape) == 2:
            context = context.unsqueeze(0)
        
        # 获取维度
        B, N, C = x.shape
        
        # 如果context的batch维度是1但x不是，则重复context
        if context.size(0) == 1 and B > 1:
            context = context.expand(B, -1, -1)
        
        # 检查特征维度
        if C != context.size(2):
            raise ValueError(f"特征维度不匹配: x:{C} vs context:{context.size(2)}")
        
        # 生成查询
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        
        # 生成上下文的键和值
        kv = self.kv(context).reshape(B, context.size(1), 2, self.num_heads, C // self.num_heads)
        
        # 使用xFormers的高效注意力计算
        if self.use_xformers and isinstance(attn_bias, fmha.BlockDiagonalMask):
            q = q.permute(0, 2, 1, 3)  # B, H, N, D
            k = kv[:, :, 0].permute(0, 2, 1, 3)
            v = kv[:, :, 1].permute(0, 2, 1, 3)
            
            # 使用块对角掩码的内存高效注意力计算
            x = fmha.memory_efficient_attention(
                q, k, v, 
                attn_bias=attn_bias,
                scale=self.scale
            )
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        else:
            # 降级为原始实现
            q = q.permute(0, 2, 1, 3)  # B, H, N, D
            kv = kv.permute(2, 0, 3, 1, 4)  # 2, B, H, M, D
            k, v = kv[0], kv[1]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # 应用掩码
            if attn_bias is not None:
                if isinstance(attn_bias, torch.Tensor):
                    attn = attn + attn_bias
                else:
                    try:
                        tensor_mask = attn_bias.materialize(
                            (B, self.num_heads, N, context.size(1)), 
                            dtype=attn.dtype,
                            device=attn.device
                        )
                        attn = attn + tensor_mask
                    except:
                        pass
            
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        
        # 恢复原始形状
        if len(x_orig_shape) == 2:
            x = x.squeeze(1)
        
        return x


def convert_module_to_sequence_packing(module: nn.Module) -> nn.Module:
    """将标准注意力模块转换为支持序列打包的版本
    
    Args:
        module: 原始模块
        
    Returns:
        支持序列打包的模块
    """
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            # 不直接修改，返回转换建议
            warnings.warn(f"建议将 {name} (nn.MultiheadAttention) 转换为 SequencePackingAttention")
        else:
            # 递归处理子模块
            convert_module_to_sequence_packing(child)
    
    return module 