##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################
import numpy as np
import torch
import torch.nn as nn
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.gridding_loss import GriddingLoss
from extensions.emd import emd_module as emd
from models.build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
import sys
import torchsummary
from torchinfo import summary
import torch.nn.functional as F
import math


class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
                #print('attn')
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
                print('rw_deform')
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
                print('deform')
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
                #print('graph')
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
                print('deformgraph')
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
                #print('attn')
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
                print('deform')
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
                #print('graph')
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
                print('deform_graph')
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

######################################## Grouper ########################################

class SelfAttentionPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionPooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Softmax(dim=1)
        )
        self.mlp = nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels

    def forward(self, x):
        # 计算注意力权重
        scores = self.score_fn(x)  # (B, n, 1)

        # 加权求和
        pooled_output = torch.sum(scores * x, dim=1)  # (B, c)

        # 应用 MLP
        output = self.mlp(pooled_output)  # (B, out_channels)
        #output = self.tanh(pooled_output)  # (B, out_channels)
        return output



class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(10, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        self.atten_pool1 = SelfAttentionPooling(256, 256)
        self.atten_pool2 = SelfAttentionPooling(512, 512)

    def forward(self, point_groups):
        '''
            point_groups : B G N 3/10
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 10)
        # Adaptive pool

        # linear attention
        feature = self.first_conv(point_groups.transpose(2,1)) # BG N 256(C)
        #feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature_global = self.atten_pool1(feature.transpose(2,1)) # # BG 256
        feature_global = feature_global.unsqueeze(2) # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1) # BG 512 n
        feature = self.second_conv(feature)  # BG 512 n
        feature_global = self.atten_pool2(feature.transpose(2,1))
        #feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 512
        return feature_global.reshape(bs, g, self.encoder_channel)
##################################################################
class SimpleEncoder(nn.Module):
    def __init__(self, k = 32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.num_features = embed_dims


    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1] 
        #n_group=256中心点，xyz 16 2048 3
        #center = misc.fps(xyz, n_group) # B G 3/ 16 256 3
        center = misc.random_sample(xyz, n_group)
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood self.group_size=32
        idx = knn_point(self.group_size, xyz, center) # 16 256 32
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points # 16 1 1
        idx = idx + idx_base
        xyz_tile = center.unsqueeze(2).repeat(1, 1, idx.shape[-1], 1) # B 256 32 3
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] #邻居点 N 3
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous() # B 256 32 3
        relative_xyz = xyz_tile - neighborhood
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # 点之间的距离
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighborhood], dim=-1)  # 连接
        assert relative_feature.size(1) == n_group
        assert relative_feature.size(2) == self.group_size
            
        features = self.embedding(relative_feature) # B G C

        return center, features

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)
        #self.attention_pooling5 = SelfAttentionPooling(384, 384)

    def forward(self, rec_feature):
        '''
        Input BNC 增广向量
        '''
        batch_size = rec_feature.size(0)
        #print("rec_feature", rec_feature.shape)
        g_feature = rec_feature.max(1)[0] #最大池化
        #g_feature = self.attention_pooling5(rec_feature)

        #print("g_feature", g_feature.shap
        token_feature = rec_feature
           # 聚合特征
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

class Top(nn.Module):
    def __init__(self, node_feature =16, encoder_feature =1024, nlevels =8 , num_pred= 2048 ):  # node_feature = 8, encoder_feature = 1024, nlevels = 8, num_pred = 2048
        super().__init__()
        self.node_feature = node_feature
        self.encoder_feature = encoder_feature  # 编码深度
        self.nlevels = nlevels  # 树的层数
        self.num_pred = num_pred  # 预测点数

        self.tree_arch = {
            2: [32, 64],
            4: [4, 8, 8, 8],
            6: [2, 4, 4, 4, 4, 4],
            8: [2, 2, 2, 2, 2, 4, 4, 4]
        }

        self.tarch = self.get_arch(self.nlevels, self.num_pred)  # 生成树状结构

        self.Top_in_channel = self.encoder_feature + self.node_feature  # 1024+8
        self.Top_out_channel = self.node_feature  # 8

        self.root_layer = nn.Sequential(
            nn.Linear(self.encoder_feature, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.node_feature * int(self.tarch[0])),  # 64
            nn.Tanh()
        )
        self.leaf_layer = self.get_tree_layer(self.Top_in_channel, 3, int(self.tarch[-1]))

        self.feature_layers = nn.ModuleList(
            [self.get_tree_layer(self.Top_in_channel, self.Top_out_channel, int(self.tarch[d])) for d in
             range(1, self.nlevels - 1)])

        # 生成树状结构
    def get_arch(self, nlevels, npts):

        logmult = int(math.log2(npts / 2048))  # 取2048的整数倍
        assert 2048 * (2 ** (logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)  # 断言2048的整数倍
        arch = self.tree_arch[nlevels]  # 树的层数
        # 调整架构
        while logmult > 0:
            last_min_pos = np.where(arch == np.min(arch))[0][-1]
            arch[last_min_pos] *= 2
            logmult -= 1
        return arch

    @staticmethod
    def get_tree_layer(in_channel, out_channel, node):
        return nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 2, 1),
            nn.BatchNorm1d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 2, in_channel // 4, 1),
            nn.BatchNorm1d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 4, in_channel // 8, 1),
            nn.BatchNorm1d(in_channel // 8),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 8, out_channel * node, 1),
        )
    def forward(self, feature_global):
        bs = feature_global.size(0)
        # decoder
        # 树状解码器
        level10 = self.root_layer(feature_global).reshape(-1, self.node_feature ,int(self.tarch[0])) # B 8 node 2
        #print("pred_coarse", level10.shape)
        outs = [level10,]  # 初始化
        for i in range(1, self.nlevels):
            last_level = outs[-1]
            expand_feature = feature_global.unsqueeze(2).expand(-1,-1,last_level.shape[2])
            if i == self.nlevels - 1:
                layer_feature = self.leaf_layer(torch.cat([expand_feature,last_level],dim=1)).reshape(bs, 3 ,-1)
            else:
                layer_feature = self.feature_layers[i-1](torch.cat([expand_feature,last_level],dim=1)).reshape(bs, self.node_feature, -1)
            outs.append(nn.Tanh()(layer_feature))
        pred = outs[-1].transpose(1,2).contiguous()
        return pred

######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config#编码 图神经网络
        decoder_config = config.decoder_config#解码 全连接层
        self.center_num  = getattr(config, 'center_num', [512, 128])#返回center_num
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3#输入通道
        self.num_query = query_num = config.num_query#512
        global_feature_dim = config.global_feature_dim#全局特征维度1024
        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        self.grouper = SimpleEncoder(k = 32, embed_dims=512)#否则使用SimpleEncoder
        #位置嵌入
        self.atten_pool3 = SelfAttentionPooling(1024, 1024)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),#3->128
            nn.GELU(),#Gaussian Error Linear Unit
            nn.Linear(128, encoder_config.embed_dim)#128->384
        )
        #输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),#grouper.num_features（聚合后特征通道）->512
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)#512->384
        )
        #################### Coarse Level 1 : Encoder ####################
        #特征融合位置编码
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        # 增加特征维度
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),#384->1024
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))#1024->1024(global_feature_dim)
        # query generator
        # (粗糙预测？)
        self.coarse_top = Top()
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),#1024->1024
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)#1024->3*512=1536
        )
        # query mlp连接层？
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),#1024+3（特征维度+xyz位置维度）->1024
            nn.GELU(),
            nn.Linear(1024, 1024),#1024->1024
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)#1024->384(decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim（判断编解码器嵌入维度相等）
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)#线性层连接编解码器
        #################### Coarse Level 2 : Decoder ####################
        self.decoder = PointTransformerDecoderEntry(decoder_config)
        #query映射排列
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),#3->256
            nn.GELU(),
            nn.Linear(256, 256),#256->256
            nn.GELU(),
            nn.Linear(256, 1),#256->1
            nn.Sigmoid()#将线性变换的结果压缩到 0 到 1 之间，表示相对于其他查询的排序概率
        )
        self.apply(self._init_weights)#初始化权重

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)#截断正态分布进行权重初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, xyz):
        bs = xyz.size(0)#输入张量 xyz 的第一个维度，batchsize
        coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)#对坐标coor进行位置编码
        x = self.input_proj(f)#对特征f进行输入投影

        x = self.encoder(x + pe, coor) # b n c将位置编码融合进特征中
        global_feature = self.increase_dim(x) # B 1024 N 将特征x进行升维为全局特征
        #print("feature_global", global_feature.shape)
#################AdaptiveMaxPool
        '''adaptive_pool = nn.AdaptiveMaxPool1d(1)
        global_feature = global_feature.permute(0,2,1)
        print("feature_global", global_feature.shape)
        

        #print("feature_global", global_feature1.shape)
        global_feature = adaptive_pool(global_feature)
        global_feature = global_feature.squeeze(dim=2)'''
######################
        #global_feature = global_feature.permute(0, 2, 1)
        global_feature = self.atten_pool3(global_feature) #  1024
        #global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        coar_top = self.coarse_top(global_feature)
        #coar_top = misc.fps(coar_top, self.num_query)
        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        coarse_inp = misc.fps(xyz, self.num_query // 2)
        #coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?
        #print("cat", coarse.shape)
        mem = self.mem_link(x)
        #### 打印coarse
        '''b, n, _ = coarse.shape
        coarse_reshaped = coarse.reshape(b * n, -1)
        coarse_cpu = coarse_reshaped.cpu().detach().numpy()
        print(coarse_cpu)
        # 将张量保存到文本文件
        file_path = "coarse.txt"
        np.savetxt(file_path, coarse_cpu, fmt="%f", delimiter=" ")'''
        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))
        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.random_sample(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64
            # produce query
            coarse_cat = torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1), coarse], dim=-1)
            q = self.mlp_query(coarse_cat)
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)
            #print("q", q.shape)
            return q, coarse, denoise_length, coar_top

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, coar_top

######################################## PoinTr ########################################  

@MODELS.register_module()
class MSPCN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.base_model = PCTransformer(config)

        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
        else:
            self.factor = self.fold_step**2
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func_l1()
        self.build_loss_func_l2()
        #self.build_loss_func_emd()
        #self.build_loss_func_grid()
        self.atten_pool4 = SelfAttentionPooling(1024, 1024)

    def build_loss_func_l1(self):
        self.loss_func_l1 = ChamferDistanceL1()
    def build_loss_func_l2(self):
        self.loss_func_l2 = ChamferDistanceL2()
    '''def build_loss_func_emd(self):
        self.loss_func_emd = emd.emdModule()
    def build_loss_func_grid(self):
        self.loss_func_grid = GriddingLoss()'''

    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine, coarse_top = ret
        gt_fps = misc.fps(gt, int(self.num_query))
        #print("gt_fps", gt_fps.shape)
        assert pred_fine.size(1) == gt.size(1) * 2
        assert pred_coarse.size(1) == gt_fps.size(1)
        #assert coarse_top.size(1) == 1/8 * gt.size(1)
        #print("coarse_top", coarse_top.shape)
        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k
        denoised_target = index_points(gt, idx) # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        #assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func_l1(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        loss_top = self.loss_func_l2(coarse_top, gt)
        loss_top = loss_top * 0.5
        # recon loss
        loss_coarse1 = self.loss_func_l2(pred_coarse, gt)
        loss_coarse2 = self.loss_func_l2(pred_coarse, gt_fps)
        loss_fine = self.loss_func_l2(pred_fine, gt)
        loss_recon =  loss_fine  + loss_coarse1 + loss_coarse2 + loss_top

        return loss_denoised, loss_recon

    def forward(self, xyz):
        q, coarse_point_cloud, denoise_length, coarse_top = self.base_model(xyz) # B M C and B M 3
        B, M ,C = q.shape
        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = self.atten_pool4(global_feature)
        '''adaptive_pool = nn.AdaptiveMaxPool1d(1)
        global_feature = global_feature.permute(0, 2, 1)
        global_feature = adaptive_pool(global_feature)
        global_feature = global_feature.squeeze(dim=2)'''
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C 将特征与相对点位置连接，得到的增广特征向量
        #print("rebuild_feature", rebuild_feature.shape)

        #global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        # NOTE: foldingNet
        rebuild_feature = self.reduce_map(rebuild_feature) # B M C线性层
        relative_xyz = self.decode_head(rebuild_feature) #
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3
        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            #print("pred_fine", pred_fine.shape)
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()
            #print("pred_coarse", pred_coarse.shape)
            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            #print("denoised_fine", denoised_fine.shape)
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()
            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query
            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine, coarse_top)

            return ret
        else:

            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3
            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query
            ret = (coarse_point_cloud, rebuild_points)

            return ret


