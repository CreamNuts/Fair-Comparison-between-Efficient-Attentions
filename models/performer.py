from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models import register_model
from torch import nn

from .base import Block, StageTransformer, _cfg_pyramid


def sample_orf(num_heads, head_dim, m):
    orf = []
    for _ in range(num_heads):
        orth = torch.empty(m, head_dim)
        torch.nn.init.orthogonal_(orth)
        orf.append(orth)
    orf = torch.stack(orf)
    return orf * np.sqrt(m)  # H M C


class Performer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        kernel_ratio=0.5,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.25

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.m = int(self.head_dim * kernel_ratio)
        self.w = nn.Parameter(
            sample_orf(num_heads, self.head_dim, self.m), requires_grad=False
        )  # H M C
        self.epsilon = 1e-6

    def kernel(self, x):
        # x = (B, H, N, C)
        # w = (H, M, C)
        # return : x : B, H, N, M
        x = x * self.scale
        x = (
            torch.einsum("BHNC,HMC->BHNM", x.float(), self.w)
            - repeat((x ** 2).sum(dim=-1), "B H N -> B H N M", M=self.m) / 2
        )
        return torch.exp(x) / np.sqrt(self.m)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H N C", qkv=3, H=self.num_heads)
        q, kt, v = (
            self.kernel(qkv[0]),
            rearrange(self.kernel(qkv[1]), "B H N M -> B H M N"),
            qkv[2],
        )
        with torch.cuda.amp.autocast(enabled=False):
            D = torch.einsum("BHNM,BHM->BHN", q, kt.sum(dim=-1))
            D = repeat(D, "B H N -> B H N C", C=self.head_dim) + self.epsilon
            ktv = torch.einsum("BHMN,BHNC->BHMC", kt, v.float())
            x = torch.einsum("BHNM,BHMC->BHNC", q, ktv) / D
        x = rearrange(x, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self):  # O(NmC) + O(NC^2)[proj]
        N = self.input_resolution[0] * self.input_resolution[1]
        # calculate flops for token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # 2 kerenl
        flops += 2 * self.num_heads * N * self.head_dim * self.m
        # ktv = torch.einsum("BHMN,BHNC->BHMC", kt, v.float())
        flops += self.num_heads * N * self.head_dim * self.m
        # x = torch.einsum("BHNM,BHMC->BHNC", q, ktv) / D
        flops += self.num_heads * N * self.head_dim * self.m
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@register_model
def stage_tiny_perf_p4(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=4, kernel_ratio=0.5, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=Performer), **cfg)
    return model


@register_model
def stage_tiny_perf_p7(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=7, kernel_ratio=0.5, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=Performer), **cfg)
    return model
