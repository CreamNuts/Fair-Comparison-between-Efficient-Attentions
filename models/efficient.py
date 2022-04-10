from functools import partial

from einops import rearrange
from timm.models import register_model
from torch import nn

from .stage import Block, StageTransformer, _cfg


class EfficientAttention(nn.Module):
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H N C", qkv=3, H=self.num_heads)
        q, kt, v = [
            qkv[0].softmax(dim=-1),  # B H N C
            rearrange(qkv[1], "B H N C -> B H C N").softmax(dim=-1),
            qkv[2],  # B H N C
        ]  # make torchscript happy (cannot use tensor as tuple)

        context = kt @ v
        context = self.attn_drop(context)

        x = rearrange(q @ context, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self): #O(NC^2)
        N = self.input_resolution[0] * self.input_resolution[1]
        # calculate flops for token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # context = kt @ v
        flops += self.num_heads * self.head_dim * N * self.head_dim
        # x = rearrange(q @ context, "B H N C -> B N (H C)")
        flops += self.num_heads * N * self.head_dim * self.head_dim
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@register_model
def stage_tiny_eff_p4(pretrained=False, **kwargs):
    cfg = _cfg(patch_size=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = StageTransformer(partial(Block, attn_layer=EfficientAttention), **cfg)
    return model


@register_model
def stage_tiny_eff_p7(pretrained=False, **kwargs):
    cfg = _cfg(patch_size=7, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = StageTransformer(partial(Block, attn_layer=EfficientAttention), **cfg)
    return model
