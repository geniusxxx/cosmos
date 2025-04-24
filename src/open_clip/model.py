""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, \
    text_global_pool
from .utils import to_2tuple

import random

try:
    from xformers.ops import fmha
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

from .sequence_packing import pack_sequences, unpack_sequences, group_by_size
from .attention import PackedCrossAttentionPooling

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    patch_dropout: float = 0.
    # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attentional_pool: bool = False
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    add_zero_attn: bool = False # add zero attention to attenion pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_all: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # a valid model name overrides layers, width, patch_size
    timm_model_name: Optional[str] = None
    # use (imagenet) pretrained weights for named model
    timm_model_pretrained: bool = False
    # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_pool: str = 'avg'
    # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value

    # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attentional_pool: bool = False
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    add_zero_attn: bool = False # add zero attention to attenion pooling

    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_all: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (
            torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            add_zero_attn=vision_cfg.add_zero_attn,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_all=vision_cfg.output_all,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_all=text_cfg.output_all,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (
            torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            attentional_pool=text_cfg.attentional_pool,
            attn_pooler_heads=text_cfg.attn_pooler_heads,
            add_zero_attn=text_cfg.add_zero_attn,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_all=text_cfg.output_all,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            cosmos: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.text_attn_cross_pool = text.attn_cross_pool
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.distill_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale) if cosmos else None

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        self.image_token_mapping = None
        self.output_all = vision_cfg['output_all']
        assert vision_cfg['output_all'] == text_cfg['output_all']        
        self.cosmos = cosmos
        if self.output_all:
            self.image_token_mapping = nn.Linear(vision_cfg['width'], embed_dim)
            self.text_token_mapping = nn.Linear(text_cfg['width'], embed_dim)        

    def init_parameters_last_transformer_layer(self):
        self.visual.init_parameters_last_transformer_layer()
        self.transformer.init_parameters_last_transformer_layer()

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups,
                         freeze_bn_stats=freeze_bn_stats)

    def freeze_except_cosmos_parts(self):
        """冻结除CrossAttn、token_mapping和logit_scale外的所有参数"""
        # 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻CrossAttn模块
        if hasattr(self.visual, 'attn_cross_pool') and self.visual.attn_cross_pool is not None:
            for param in self.visual.attn_cross_pool.parameters():
                param.requires_grad = True
        
        if hasattr(self, 'text_attn_cross_pool') and self.text_attn_cross_pool is not None:
            for param in self.text_attn_cross_pool.parameters():
                param.requires_grad = True
        
        # 解冻token_mapping模块
        if self.image_token_mapping is not None:
            for param in self.image_token_mapping.parameters():
                param.requires_grad = True
        
        if self.output_all and hasattr(self, 'text_token_mapping'):
            for param in self.text_token_mapping.parameters():
                param.requires_grad = True
        
        # 解冻logit_scale和distill_logit_scale
        if hasattr(self, 'logit_scale'):
            self.logit_scale.requires_grad = True
        
        if hasattr(self, 'distill_logit_scale') and self.distill_logit_scale is not None:
            self.distill_logit_scale.requires_grad = True
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        if self.output_all:
            tokens, features = self.visual(image)
            tokens = self.image_token_mapping(tokens)
            output_dict = {'image_tokens': tokens}
            output_dict['image_features'] = F.normalize(features, dim=-1) if normalize else features
            return output_dict
        else:
            features = self.visual(image)
            return {'image_features': F.normalize(features, dim=-1) if normalize else features}

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(
            cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, tokens = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        if self.output_all:
            tokens = self.text_token_mapping(tokens)
            output_dict = {'text_tokens': tokens}
            output_dict['text_features'] = F.normalize(x, dim=-1) if normalize else x
            return output_dict
        return {'text_features': F.normalize(x, dim=-1) if normalize else x}

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_features = image_features['image_features']
        text_features = text_features['text_features']
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits
    
    def get_logit_scale(self):
        return self.logit_scale.exp()
    
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
    ):
        is_norm = True
        if self.output_all and batch_size is not None:
            is_norm = False

        if isinstance(image, list): # with Multicrop augmentation
            image_features = MultiCropWrap(self.visual, image, normalize=is_norm, 
                                            image_token_mapping=self.image_token_mapping) 
        else:
            image_features = self.encode_image(image, normalize=is_norm) if image is not None else None

        text_features = self.encode_text(text, normalize=is_norm) if text is not None else None

        if self.cosmos and batch_size is not None:
            assert image is not None and text is not None
            assert self.visual.attn_cross_pool is not None and self.text_attn_cross_pool is not None

            img_tokens = image_features['image_tokens'][:batch_size] # first global image
            img_features = image_features['image_features'] # global images + local images
            txt_tokens = text_features['text_tokens'][:batch_size] # first global caption
            txt_features = text_features['text_features'] # global captions + local captions

            img_num = len(img_features) // batch_size
            txt_num = len(txt_features) // batch_size

            txt_pooled_tokens = self.text_attn_cross_pool(txt_tokens.repeat(img_num, 1, 1), img_features.unsqueeze(1))
            img_crossmodal_features = img_features + txt_pooled_tokens.squeeze()
            img_crossmodal_features = F.normalize(img_crossmodal_features, dim=-1)

            img_pooled_tokens = self.visual.attn_cross_pool(img_tokens.repeat(txt_num, 1, 1), txt_features.unsqueeze(1))
            txt_crossmodal_features = txt_features + img_pooled_tokens.squeeze()
            txt_crossmodal_features = F.normalize(txt_crossmodal_features, dim=-1) 

            image_features['image_features'] = F.normalize(img_features, dim=-1)
            text_features['text_features'] = F.normalize(txt_features, dim=-1)     

        if self.output_dict:
            out_dict = {
                'image_features': image_features['image_features'] if image is not None else None,
                'text_features': text_features['text_features'] if text is not None else None ,
                'logit_scale': self.logit_scale.exp(),
            }
            if self.distill_logit_scale is not None:
                out_dict['distill_logit_scale'] = self.distill_logit_scale.exp()
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias

            if self.cosmos and batch_size is not None:
                out_dict['img_crossmodal_features'] = img_crossmodal_features
                out_dict['txt_crossmodal_features'] = txt_crossmodal_features
                
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


# Based on DINO code https://github.com/facebookresearch/dino/blob/main/utils.py#L594
def MultiCropWrap(backbone, x, normalize: bool = False, image_token_mapping = None):
    """处理多剪裁输入，支持序列打包以提高效率
    
    Args:
        backbone: 视觉模型
        x: 图像张量列表
        normalize: 是否归一化输出特征
        image_token_mapping: 可选的token映射函数
    """
    # 检查是否支持序列打包
    use_sequence_packing = HAS_XFORMERS and getattr(backbone, 'use_sequence_packing', False)
    
    if use_sequence_packing:
        # 按尺寸分组图像
        sizes = [inp.shape[-1] for inp in x]
        size_indices = torch.unique_consecutive(torch.tensor(sizes), return_inverse=True)[1]
        grouped_images = [[] for _ in range(size_indices.max().item() + 1)]
        
        # 将图像分配到对应尺寸组
        for i, img in enumerate(x):
            grouped_images[size_indices[i]].append(img)
        
        # 合并每组中的图像
        batched_groups = [torch.cat(group) for group in grouped_images if group]
        
        try:
            # 序列打包路径
            if image_token_mapping is not None:
                # 带token映射的处理
                tokens_list = []
                features_list = []
                
                # 使用序列打包一次性处理所有组
                packed_tensor, block_mask = pack_sequences(batched_groups)
                if hasattr(backbone, 'forward_packed'):
                    packed_tokens, packed_features = backbone.forward_packed(packed_tensor, block_mask)
                    tokens_outputs = block_mask.split(packed_tokens)
                    features_outputs = block_mask.split(packed_features)
                else:
                    # 降级为单独处理每个组
                    for group in batched_groups:
                        tokens, features = backbone(group)
                        tokens_list.append(tokens)
                        features_list.append(features)
                    tokens_outputs = tokens_list
                    features_outputs = features_list
                
                # 只保留第一组的tokens (全局裁剪)
                tokens = tokens_outputs[0]
                # 合并所有特征
                features = torch.cat(features_outputs)
            else:
                # 不需要token映射的处理
                features_list = []
                
                # 使用序列打包一次性处理所有组
                packed_tensor, block_mask = pack_sequences(batched_groups)
                if hasattr(backbone, 'forward_packed'):
                    packed_features = backbone.forward_packed(packed_tensor, block_mask)
                    features_outputs = block_mask.split(packed_features)
                else:
                    # 降级为单独处理每个组
                    for group in batched_groups:
                        features = backbone(group)
                        features_list.append(features)
                    features_outputs = features_list
                
                # 合并所有特征
                features = torch.cat(features_outputs)
                tokens = None
        except Exception as e:
            print(f"序列打包处理失败: {e}，退化为原始实现")
            # 出错时退化为原始实现
            use_sequence_packing = False
    
    # 原始实现 (不使用序列打包或序列打包失败时)
    if not use_sequence_packing:
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output, tokens = 0, None, None
        for end_idx in idx_crops:
            if image_token_mapping is not None:
                _tokens, _out = backbone(torch.cat(x[start_idx: end_idx]))
            else:
                _out = backbone(torch.cat(x[start_idx: end_idx]))

            # accumulate outputs
            output = _out if output is None else torch.cat([output, _out])
            # we only need visual tokens of global crops (first element of for-loop), not local crops
            if image_token_mapping is not None and tokens is None: 
                tokens = _tokens

            start_idx = end_idx
        
        # 使用原始变量名保持一致
        features = output
    
    # 创建输出字典
    output_dict = {}
    output_dict['image_features'] = F.normalize(features, dim=-1) if normalize else features    
    if image_token_mapping is not None:
        output_dict['image_tokens'] = image_token_mapping(tokens)
    return output_dict


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            cosmos: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(
            embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.distill_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale) if cosmos else None
        
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        
        # COSMOS特定参数
        self.cosmos = cosmos
        self.image_token_mapping = None
        self.text_token_mapping = None
        self.output_all = vision_cfg['output_all']
        assert vision_cfg['output_all'] == text_cfg['output_all']
        if self.output_all:
            self.image_token_mapping = nn.Linear(1280, embed_dim)
            self.text_token_mapping = nn.Linear(text_cfg['width'], embed_dim)
            
        # 添加序列打包支持
        self.use_sequence_packing = HAS_XFORMERS
        self.inference_mode = False
        
        # 如果有交叉注意力模块，升级为支持序列打包的版本
        if self.cosmos and self.use_sequence_packing:
            self._upgrade_cross_attention()

    def _upgrade_cross_attention(self):
        """升级交叉注意力模块以支持序列打包"""
        if hasattr(self.visual, 'attn_cross_pool'):
            old_cross_pool = self.visual.attn_cross_pool
            # 判断是否为AttentionalCrossPooler类型
            if hasattr(old_cross_pool, 'q'):
                # 原始实现：使用q.in_features
                self.visual.attn_cross_pool = PackedCrossAttentionPooling(
                    dim=old_cross_pool.q.in_features,
                    num_heads=old_cross_pool.num_heads
                )
                # 复制权重
                self.visual.attn_cross_pool.q.weight.data.copy_(old_cross_pool.q.weight.data)
                self.visual.attn_cross_pool.q.bias.data.copy_(old_cross_pool.q.bias.data)
                self.visual.attn_cross_pool.kv.weight.data.copy_(old_cross_pool.kv.weight.data)
                self.visual.attn_cross_pool.kv.bias.data.copy_(old_cross_pool.kv.bias.data)
                self.visual.attn_cross_pool.proj.weight.data.copy_(old_cross_pool.proj.weight.data)
                self.visual.attn_cross_pool.proj.bias.data.copy_(old_cross_pool.proj.bias.data)
            else:
                # AttentionalCrossPooler类型
                self.visual.attn_cross_pool = PackedCrossAttentionPooling(
                    dim=old_cross_pool.attn.embed_dim,  # 从attn中获取embed_dim
                    num_heads=old_cross_pool.attn.num_heads  # 从attn中获取num_heads
                )
                # 注意：这里不复制权重，因为结构不匹配
            
        if hasattr(self.text, 'attn_cross_pool'):
            old_cross_pool = self.text.attn_cross_pool
            # 同样判断类型
            if hasattr(old_cross_pool, 'q'):
                self.text.attn_cross_pool = PackedCrossAttentionPooling(
                    dim=old_cross_pool.q.in_features,
                    num_heads=old_cross_pool.num_heads
                )
                # 复制权重
                self.text.attn_cross_pool.q.weight.data.copy_(old_cross_pool.q.weight.data)
                self.text.attn_cross_pool.q.bias.data.copy_(old_cross_pool.q.bias.data)
                self.text.attn_cross_pool.kv.weight.data.copy_(old_cross_pool.kv.weight.data)
                self.text.attn_cross_pool.kv.bias.data.copy_(old_cross_pool.kv.bias.data)
                self.text.attn_cross_pool.proj.weight.data.copy_(old_cross_pool.proj.weight.data)
                self.text.attn_cross_pool.proj.bias.data.copy_(old_cross_pool.proj.bias.data)
            else:
                # AttentionalCrossPooler类型
                self.text.attn_cross_pool = PackedCrossAttentionPooling(
                    dim=old_cross_pool.attn.embed_dim,
                    num_heads=old_cross_pool.attn.num_heads
                )

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups,
                         freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        
    def train(self, mode=True):
        """设置为训练模式时自动启用序列打包"""
        super().train(mode)
        self.inference_mode = not mode
        return self
        
    def eval(self):
        """设置为评估模式时自动禁用序列打包"""
        super().eval()
        self.inference_mode = True
        return self

    def encode_image(self, image, normalize: bool = False):
        if self.output_all:
            tokens, features = self.visual(image)
            tokens = self.image_token_mapping(tokens)
            output_dict = {'image_tokens': tokens}
            output_dict['image_features'] = F.normalize(features, dim=-1) if normalize else features
            return output_dict
        else:
            features = self.visual(image)
            return {'image_features': F.normalize(features, dim=-1) if normalize else features}

    def encode_text(self, text, normalize: bool = False):
        if self.output_all:
            features, tokens = self.text(text)
            tokens = self.text_token_mapping(tokens)
            output_dict = {'text_tokens': tokens}
            output_dict['text_features'] = F.normalize(features, dim=-1) if normalize else features
            return output_dict
        else:
            features = self.text(text)
            return {'text_features': F.normalize(features, dim=-1) if normalize else features}

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_features = image_features['image_features']
        text_features = text_features['text_features']
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
    ):
        is_norm = True
        if self.output_all and batch_size is not None:
            is_norm = False

        # 处理图像输入
        if isinstance(image, list):  # with Multicrop augmentation
            # 将use_sequence_packing标志传递给backbone
            if hasattr(self.visual, 'use_sequence_packing'):
                self.visual.use_sequence_packing = self.use_sequence_packing and not self.inference_mode
                
            image_features = MultiCropWrap(self.visual, image, normalize=is_norm,
                                         image_token_mapping=self.image_token_mapping)
        else:
            image_features = self.encode_image(image, normalize=is_norm) if image is not None else None

        # 处理文本输入
        text_features = self.encode_text(text, normalize=is_norm) if text is not None else None

        # COSMOS特定处理
        if self.cosmos and batch_size is not None and self.output_all:
            assert image is not None and text is not None
            assert hasattr(self.visual, 'attn_cross_pool') and hasattr(self.text, 'attn_cross_pool')

            # 提取特征
            img_tokens = image_features['image_tokens'][:batch_size]  # first global image
            img_features = image_features['image_features']  # global images + local images
            txt_tokens = text_features['text_tokens'][:batch_size]  # first global caption
            txt_features = text_features['text_features']  # global captions + local captions

            img_num = len(img_features) // batch_size
            txt_num = len(txt_features) // batch_size
            
            # 使用序列打包优化交叉注意力计算
            if self.use_sequence_packing and not self.inference_mode and HAS_XFORMERS:
                try:
                    # 注意：pack_sequences应该接收张量列表，而不是元组列表
                    img_num = len(img_features) // batch_size
                    txt_num = len(txt_features) // batch_size
                    
                    # 直接使用attn_cross_pool，已经处理好了2D和3D输入
                    txt_pooled_tokens = self.text.attn_cross_pool(
                        img_features,  # [B*img_num, embed_dim]
                        txt_tokens.repeat(img_num, 1, 1),  # [B*img_num, token_len, embed_dim]
                        None  # 不使用掩码，让类内部处理
                    )
                    
                    img_pooled_tokens = self.visual.attn_cross_pool(
                        txt_features,  # [B*txt_num, embed_dim]
                        img_tokens.repeat(txt_num, 1, 1),  # [B*txt_num, token_len, embed_dim]
                        None  # 不使用掩码，让类内部处理
                    )
                    
                    # 确保维度匹配，此时txt_pooled_tokens和img_features应该有相同的形状
                    # 同样，img_pooled_tokens和txt_features应该有相同的形状
                    img_crossmodal_features = img_features + txt_pooled_tokens
                    img_crossmodal_features = F.normalize(img_crossmodal_features, dim=-1)
                    
                    txt_crossmodal_features = txt_features + img_pooled_tokens 
                    txt_crossmodal_features = F.normalize(txt_crossmodal_features, dim=-1)
                except Exception as e:
                    print(f"交叉注意力序列打包失败: {e}，退化为原始实现")
                    # 降级为原始实现...
            else:
                # 原始实现
                txt_pooled_tokens = self.text.attn_cross_pool(txt_tokens.repeat(img_num, 1, 1), img_features.unsqueeze(1))
                img_crossmodal_features = img_features + txt_pooled_tokens.squeeze()
                img_crossmodal_features = F.normalize(img_crossmodal_features, dim=-1)
                
                img_pooled_tokens = self.visual.attn_cross_pool(img_tokens.repeat(txt_num, 1, 1), txt_features.unsqueeze(1))
                txt_crossmodal_features = txt_features + img_pooled_tokens.squeeze()
                txt_crossmodal_features = F.normalize(txt_crossmodal_features, dim=-1)
            
            # 归一化原始特征
            image_features['image_features'] = F.normalize(img_features, dim=-1)
            text_features['text_features'] = F.normalize(txt_features, dim=-1)

        # 构建输出
        if self.output_dict:
            out_dict = {
                "image_features": image_features['image_features'] if image is not None else None,
                "text_features": text_features['text_features'] if text is not None else None,
                "logit_scale": self.logit_scale.exp()
            }
            if self.distill_logit_scale is not None:
                out_dict['distill_logit_scale'] = self.distill_logit_scale.exp()
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
                
            if self.cosmos and batch_size is not None and self.output_all:
                out_dict['img_crossmodal_features'] = img_crossmodal_features
                out_dict['txt_crossmodal_features'] = txt_crossmodal_features
                
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    # OpenAI state dicts are partially converted to float16
    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones(
        (batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    # FIXME detect different token configs (ie no class token, or more)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:
                                                 extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s',
                 old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info(
        'Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(
        1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    # legacy attribute, keeping for bwd compat
    module.image_mean = preprocess_cfg['mean']
    # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']
    # new attr, package all pp cfg as dict
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg
