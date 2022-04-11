from functools import partial

from einops import rearrange
from timm.models import register_model
from torch import nn

from .base import (
    Block,
    ColumnarTransformer,
    StageTransformer,
    _cfg_columnar,
    _cfg_pyramid,
)


class Attention(nn.Module):
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
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, N, C = x.shape
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H N C", qkv=3, H=self.num_heads)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ rearrange(k, "B H N C -> B H C N")) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = rearrange(attn @ v, "B H N C -> B N (H C)")
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self):
        N = self.input_resolution[0] * self.input_resolution[1]
        # calculate flops for token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ rearrange(k, "B H N C -> B H C N"))
        flops += self.num_heads * N * self.head_dim * N
        # x = rearrange(attn @ v, "B H N C -> B N (H C)")
        flops += self.num_heads * N * N * self.head_dim
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@register_model
def stage_tiny_p4(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=4, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model


@register_model
def stage_tiny_p7(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=7, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model


@register_model
def column_small_p4(pretrained=False, **kwargs):
    cfg = _cfg_columnar(patch_size=4, **kwargs)
    model = ColumnarTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model


@register_model
def column_small_p7(pretrained=False, **kwargs):
    cfg = _cfg_columnar(patch_size=7, **kwargs)
    model = ColumnarTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model


@register_model
def column_small_p14(pretrained=False, **kwargs):
    cfg = _cfg_columnar(patch_size=14, **kwargs)
    model = ColumnarTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model


@register_model
def column_small_p16(pretrained=False, **kwargs):
    cfg = _cfg_columnar(patch_size=16, **kwargs)
    model = ColumnarTransformer(partial(Block, attn_layer=Attention), **cfg)
    return model
