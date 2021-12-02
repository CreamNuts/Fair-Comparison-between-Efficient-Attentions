from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models import register_model
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch import nn

from .stage import StageTransformer, _cfg


def sample_orf(num_heads, head_dim, m):
    orf = []
    for _ in range(num_heads):
        orth = torch.empty(head_dim, m)
        torch.nn.init.orthogonal_(orth)
        orf.append(orth)
    orf = torch.cat(orf, 0)
    return orf * np.sqrt(m)  # H C M


class Performer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        kernel_ratio=0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.25
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.m = int(dim * kernel_ratio)
        self.w = nn.Parameter(
            sample_orf(num_heads, head_dim, self.m), requires_grad=False
        )  # H C M
        self.epsilon = 1e-8

    def kernel(self, x):
        # x = (B, H, N, C)
        # w = (H, C, M)
        # return : x : B, H, N, M
        x = x * self.scale
        x = (
            torch.einsum("bhnc,hcm->bhnm", x, self.w)
            - repeat((x ** 2).sum(dim=-1), "1 -> 1 1 1 m", m=self.m) / 2
        )
        return torch.exp(x) / np.sqrt(self.m)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H N C", qkv=3, H=self.num_heads)
        q, k, v = (
            self.kernel(qkv[0]),
            rearrange(self.kernel(qkv[1]), "B H N M -> B H M N"),
            qkv[2],
        )
        D = torch.einsum("BHNM,BHM->BHN", q, k.sum(dim=-1))
        D = repeat(D, "B H N -> B H N C", C=self.head_dim) + self.epsilon
        ktv = torch.einsum("BHMN,BHNC->BHMC", k, v)
        x = torch.einsum("BHNM,BHMC->BHNC", q, ktv) / D
        x = rearrange(x, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerformerBlock(nn.Module):
    r"""Performer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        num_tokens = input_resolution[0] * input_resolution[1]
        self.norm1 = norm_layer(dim)
        self.attn = Performer(
            dim,
            num_tokens,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution},"
            f"num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # FIXME: attn
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


@register_model
def stage_tiny_perf_p4(pretrained=False, **kwargs):
    cfg = _cfg(
        patch_size=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), kernel_ratio=0.5, **kwargs
    )
    model = StageTransformer(PerformerBlock, **cfg)
    return model


@register_model
def stage_tiny_perf_p7(pretrained=False, **kwargs):
    cfg = _cfg(
        patch_size=7, norm_layer=partial(nn.LayerNorm, eps=1e-6), kernel_ratio=0.5, **kwargs
    )
    model = StageTransformer(PerformerBlock, **cfg)
    return model
