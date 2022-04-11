from functools import partial

from einops import rearrange
from timm.models import register_model
from torch import nn

from .base import Block, StageTransformer, _cfg_pyramid


class LinAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        kv_tokens_ratio=4,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.kv_tokens_ratio = kv_tokens_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_kv = nn.Linear(
            input_resolution[0] * input_resolution[1],
            input_resolution[0] * input_resolution[1] // kv_tokens_ratio,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, N, C = x.shape
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H C N", qkv=3, H=self.num_heads)
        q, kt, v = [
            rearrange(qkv[0], "B H C N -> B H N C"),
            self.proj_kv(qkv[1]),  # B H C K
            rearrange(self.proj_kv(qkv[2]), "B H C K -> B H K C"),
        ]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ kt) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = rearrange(attn @ v, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self):
        N = self.input_resolution[0] * self.input_resolution[1]
        # calculate flops for token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # self.proj_kv(qkv[1]), self.proj_kv(qkv[2])
        flops += 2 * self.num_heads * self.head_dim * N * N // self.kv_tokens_ratio
        # attn = (q @ kt)
        flops += self.num_heads * N * self.head_dim * N // self.kv_tokens_ratio
        # x = rearrange(attn @ v, "B H N C -> B N (H C)")
        flops += self.num_heads * N * N // self.kv_tokens_ratio * self.head_dim
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@register_model
def stage_tiny_lin_p4(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=4, kv_tokens_ratio=8, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=LinAttention), **cfg)
    return model


@register_model
def stage_tiny_lin_p7(pretrained=False, **kwargs):
    cfg = _cfg_pyramid(patch_size=7, kv_tokens_ratio=8, **kwargs)
    model = StageTransformer(partial(Block, attn_layer=LinAttention), **cfg)
    return model
