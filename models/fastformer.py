from functools import partial

import torch
from einops import rearrange, reduce
from timm.models import register_model
from torch import nn

from .base import Block, StageTransformer, _cfg_pyramid


class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.w_q = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.w_k = nn.Parameter(torch.randn(num_heads, self.head_dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H N C", qkv=3, H=self.num_heads)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        alpha = torch.einsum("BHNC,HC->BHN", q, self.w_q) * self.scale
        alpha = alpha.softmax(dim=-1)
        global_q = torch.einsum("BHN,BHNC->BHNC", alpha, q)
        global_q = reduce(global_q, "B H N C -> B H C", "sum")

        p = torch.einsum("BHC,BHNC -> BHNC", global_q, k)
        p = self.attn_drop(p)
        beta = torch.einsum("BHNC,HC->BHN", p, self.w_k) * self.scale
        beta = beta.softmax(dim=-1)
        global_k = torch.einsum("BHN,BHNC->BHNC", beta, k)
        global_k = reduce(global_k, "B H N C -> B H C", "sum")

        u = torch.einsum("BHC,BHNC->BHNC", global_k, v)

        q = rearrange(q, "B H N C -> B N (H C)")
        u = rearrange(u, "B H N C -> B N (H C)")
        x = self.proj(u) + q
        x = self.proj_drop(x)
        return x

    def flops(self):
        return NotImplementedError


@register_model
def stage_tiny_fast_p4(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=4, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=FastAttention), **cfg)
    return model


@register_model
def stage_tiny_fast_p7(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=7, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=FastAttention), **cfg)
    return model
