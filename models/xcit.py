"""
Implementation of Cross-Covariance Image Transformer (XCiT)
Based on timm and DeiT code bases
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
"""
from functools import partial

import torch
from einops import rearrange
from timm.models import register_model
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch import nn

from .stage import StageTransformer, _cfg


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        input_resolution=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        kernel_size=3,
    ):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2
        self.H, self.W = input_resolution
        self.conv1 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )

    def forward(self, x):
        x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = rearrange(x, "B C H W -> B (H W) C", H=self.H, W=self.W)
        return x


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
    sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "B N (qkv H C) -> qkv B H C N", qkv=3, H=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = rearrange((attn @ v), "B H C N -> B N (H C)")
        # x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class XCABlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        lpi_flag=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.lpi_flag = lpi_flag
        if lpi_flag:
            self.local_mp = LPI(
                in_features=dim, act_layer=act_layer, input_resolution=input_resolution
            )
            self.norm3 = norm_layer(dim)
            self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.lpi_flag:
            x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


@register_model
def stage_tiny_xcit_p4(pretrained=False, **kwargs):
    cfg = _cfg(patch_size=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, **kwargs)
    model = StageTransformer(XCABlock, **cfg)
    return model


@register_model
def stage_tiny_xcit_p7(pretrained=False, **kwargs):
    cfg = _cfg(patch_size=7, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, **kwargs)
    model = StageTransformer(XCABlock, **cfg)
    return model


@register_model
def stage_tiny_xcit_p7_no_lpi(pretrained=False, **kwargs):
    cfg = _cfg(
        patch_size=7, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, lpi_flag=False, **kwargs
    )
    model = StageTransformer(XCABlock, **cfg)
    return model
