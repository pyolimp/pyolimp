# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import matplotlib.pyplot as plt
import numpy as np


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index", relative_position_index
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
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
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(H,W,  B,L,C)
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    self.reduction = nn.Linear(
        scale * scale * dim, 0.5 * scale * scale * dim, bias=False
    )
    return x, H, W


class Upsample_promotion_interpolate(nn.Module):
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=2)
        # print(dim, dim/4)
        self.norm = norm_layer(int(dim))
        self.promotion = nn.Linear(int(dim), int(dim / 2), bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        # x1 = x.permute(0, 2, 1)
        # x1 = x1.view(-1, C, H, W)
        # x1 = nn.PixelShuffle(2)(x1) # C/4  H*2 W*2
        #
        # B, C, H, W = x1.size()
        # x1 = x1.view(-1, C, H * W)
        # x1 = x1.permute(0, 2, 1)
        # x1 = self.norm(x1)
        # x1 = self.promotion(x1)
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = self.upsample(x)
        # x = nn.functional.interpolate(input=x,)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        # x = self.promotion(x)

        return x


class Upsample_promotion(nn.Module):
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, input_resolution, dim, norm_layer=nn.LayerNorm, norm_flag=1
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # print(dim, dim/4)
        self.norm_flag = norm_flag
        self.norm = norm_layer(int(dim / 2))
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)
        self.layer = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        # x1 = x.permute(0, 2, 1)
        # x1 = x1.view(-1, C, H, W)
        # x1 = nn.PixelShuffle(2)(x1) # C/4  H*2 W*2
        #
        # B, C, H, W = x1.size()
        # x1 = x1.view(-1, C, H * W)
        # x1 = x1.permute(0, 2, 1)
        # x1 = self.norm(x1)
        # x1 = self.promotion(x1)
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)

        x = self.layer(x)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        # print('up',x.size())
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        if self.norm_flag == 1:
            x = self.norm(x)
        # x = self.promotion(x)

        return x


class Upsample(nn.Module):
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.layer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.norm = norm_layer(int(dim / 4))
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = self.layer(x)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        x = self.norm(x)

        # x1 = self.promotion(x1)
        return x


class Upsample_layer(nn.Module):
    # nn.Conv2d(dim, dim, 3, 1, 1), to
    # nn.Conv2d(dim, 2*dim, 3, 1, 1),
    # nn.Conv2d(dim, 4*dim, 3, 1, 1),
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, input_resolution, dim, output_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # print(dim, dim/4)
        self.output_dim = output_dim
        self.norm = norm_layer(int(output_dim))
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)
        self.layer = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(4 * dim, output_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        # x1 = x.permute(0, 2, 1)
        # x1 = x1.view(-1, C, H, W)
        # x1 = nn.PixelShuffle(2)(x1) # C/4  H*2 W*2
        #
        # B, C, H, W = x1.size()
        # x1 = x1.view(-1, C, H * W)
        # x1 = x1.permute(0, 2, 1)
        # x1 = self.norm(x1)
        # x1 = self.promotion(x1)
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = self.layer(x)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        # print('up',x.size())
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        # x = self.promotion(x)

        return x


class resi_connection_layer(nn.Module):
    r""" " resi_connection_layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # print(dim, dim/4)
        self.output_dim = output_dim
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)
        self.layer = nn.Sequential(
            nn.Conv2d(dim, output_dim, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(H,W,B,L,C)
        # print(self.dim,self.output_dim)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        # x1 = x.permute(0, 2, 1)
        # x1 = x1.view(-1, C, H, W)
        # x1 = nn.PixelShuffle(2)(x1) # C/4  H*2 W*2
        #
        # B, C, H, W = x1.size()
        # x1 = x1.view(-1, C, H * W)
        # x1 = x1.permute(0, 2, 1)
        # x1 = self.norm(x1)
        # x1 = self.promotion(x1)
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        # print('up', x.size())
        # print(x)
        x = self.layer(x)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        # print('up',x.size())
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        # x = self.norm(x)
        # x = self.promotion(x)

        return x


class PatchMerging_scale(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, input_resolution, dim, norm_layer=nn.LayerNorm, scale=2
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(
            scale * scale * dim, 0.5 * scale * scale * dim, bias=False
        )
        self.norm = norm_layer(scale * scale * dim)
        self.scale = scale

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert (
            H % self.scale == 0 and W % self.scale == 0
        ), f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        for i in range(self.scale):
            for j in range(self.scale):
                x0 = x[:, j :: self.scale, i :: self.scale, :]
                if (i == 0) and (j == 0):
                    tempx = x0
                else:
                    tempx = torch.cat([tempx, x0], -1)
        x = tempx  # B H/scale   W/scale  scale*scale*C
        x = x.view(B, -1, self.scale * self.scale * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"
    #
    # def flops(self):
    #     H, W = self.input_resolution
    #     flops = H * W * self.dim
    #     flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
    #     return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging_cnn(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # x  = x.view(-1,C,H*W)
        x = x.permute(0, 2, 3, 1)
        # x = x.view(-1,H,W,C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        # print(x.shape)

        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Upsample_promotion_cnn(nn.Module):
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        # print(dim, dim/4)
        # self.norm = norm_layer(int(dim / 2))
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)
        self.layer = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, 1, 1),
            nn.PixelShuffle(2),
            norm_layer(int(dim / 2)),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.layer(x)
        # x = nn.PixelShuffle(2)(x)  # C/4  H*2 W*2

        B, C, H, W = x.size()
        # print('up',x.size())
        # x = x.view(-1, C, H * W)
        # x = x.permute(0, 2, 1)
        # x = self.norm(x)
        # x = self.promotion(x)

        return x


class Upsample_cnn(nn.Module):
    r""" " upsample channels.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.layer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PixelShuffle(2),
            norm_layer(int(dim / 4)),
        )
        # self.norm = norm_layer(int(dim / 4))
        # self.promotion = nn.Linear(int(dim / 4), int(dim / 2), bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print(x.size(),'x3')
        x = self.layer(x)
        # x = self.norm(x)

        # x1 = self.promotion(x1)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):

        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=int(embed_dim // 2),
                norm_layer=norm_layer,
            ),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(24, 3, 3, padding=1),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = []
        for layer in self.layers:
            x = layer(x)

            self.downsample_result.append(x)
        i = 0

        for xx in self.downsample_result:
            i = i + 1
        x1 = x

        i = 0
        for uplayer in self.uplayers:

            x1 = uplayer(x1)

            if i < 2:
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        x = self.final_upsample(x)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 24, 256, 256])

        x = self.final(x)

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch2(nn.Module):
    r"""patch size is 2."""

    r""" Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 2
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=2,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[4, 6],
        num_heads=[3, 6],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim // 2), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Linear(int(embed_dim), 3, bias=False),
            # nn.Conv2d(int(embed_dim/2), 3, 3, padding=1),
            # nn.Conv2d()
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            # nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
            self.downsample_result.append(x)
        i = 0

        for xx in self.downsample_result:
            i = i + 1
        x1 = x

        i = 0
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 1:
                x1 = torch.cat((x1, self.downsample_result[0 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())
        x = self.final_upsample(x)
        x = self.final(x)

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.size())
        x = x.view([-1, 3, 256, 256])

        # x = self.final(x)

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch2_1_1(nn.Module):
    r"""patch size is 2."""

    r""" Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 2
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=2,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[4, 6],
        num_heads=[3, 6],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim // 2), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Linear(int(embed_dim), 30, bias=False),
            nn.Linear(30, 3, bias=False),
            # nn.Conv2d(int(embed_dim // 2), 3, 1, padding=0),
            # nn.Conv2d(int(embed_dim/2), 3, 3, padding=1),
            # nn.Conv2d()
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
            self.downsample_result.append(x)
        i = 0

        for xx in self.downsample_result:
            i = i + 1
        x1 = x

        i = 0
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 1:
                x1 = torch.cat((x1, self.downsample_result[0 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())
        x = self.final_upsample(x)
        x = self.final(x)

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.size())
        x = x.view([-1, 3, 256, 256])

        # x = self.final(x)

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_1_1(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[4, 6],
        num_heads=[3, 6],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=int(embed_dim),
                norm_layer=norm_layer,
            ),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            # nn.Conv2d(24, 3, 3, padding=1),
            # nn.Linear(48, 3, bias=False),
            nn.Linear(48, 10, bias=False),
            nn.Linear(10, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 1:
                x1 = torch.cat((x1, self.downsample_result[0 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())
        x = self.final_upsample(x)
        x = self.final(x)

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.size())
        x = x.view([-1, 3, 256, 256])

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_8_3_48_3(nn.Module):
    # cvd_100_001_D_labonlyG+global_con+nature_points1000_weightad_k_100_COLOR_patch4_unt_48*3_min0.1
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[6, 6, 6],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.flinal_layer = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[6, 6, 6],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.flinal_layer = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                output_dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
                norm_layer=norm_layer,
            ),
        )

        # self.flinal_layer

        # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
        #                    dim=embed_dim*2, norm_layer=norm_layer),
        # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
        #                    dim=int(embed_dim), norm_layer=norm_layer),

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[6, 6, 6],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:

                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.flinal_layer = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                output_dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
                norm_layer=norm_layer,
            ),
        )

        # self.flinal_layer

        # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
        #                    dim=embed_dim*2, norm_layer=norm_layer),
        # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
        #                    dim=int(embed_dim), norm_layer=norm_layer),

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_6_3_48_3(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[6, 6, 6],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                output_dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
                norm_layer=norm_layer,
            ),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                x1 = torch.cat(
                    (
                        x1,
                        self.resi_connection[1 - i](
                            self.downsample_result[1 - i]
                        ),
                    ),
                    -1,
                )
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3_server5(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=embed_dim, norm_layer=norm_layer),
            #
            # Upsample_layer(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, output_dim =embed_dim, norm_layer=norm_layer),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
                norm_layer=norm_layer,
            ),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3_nouplayer_server5(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            #
            # Upsample_layer(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, output_dim =embed_dim, norm_layer=norm_layer),
            # Upsample_layer(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                dim=embed_dim, output_dim=embed_dim // 2, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton(
    nn.Module
):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            # nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            # nn.LeakyReLU(inplace=True),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
                norm_flag=0,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
                norm_flag=0,
            ),
            #
            # Upsample_layer(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, output_dim =embed_dim, norm_layer=norm_layer),
            # Upsample_layer(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                dim=embed_dim, output_dim=embed_dim // 2, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class my_cnn(nn.Module):
    def __init__(
        self, in_size, out_size, normalize=True, dropout=0.0, downsample=None
    ):
        super(my_cnn, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if downsample != None:
            self.downsample = downsample(
                dim=in_size, norm_layer=nn.BatchNorm2d
            )
        else:
            self.downsample = None
            # layers.append(self.downsample)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size(),'1')
        x = self.model(x)
        # print(x.size(),'2')
        if self.downsample != None:
            x = self.downsample(x)
        # x = torch.cat((x, skip_input), 1)
        return x


class my_cnn_multi(nn.Module):
    def __init__(
        self, in_size, out_size, normalize=True, dropout=0.0, downsample=None
    ):
        super(my_cnn_multi, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if downsample != None:
            self.downsample = downsample(
                dim=in_size, norm_layer=nn.BatchNorm2d
            )
        else:
            self.downsample = None
            # layers.append(self.downsample)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size(),'1')
        x = self.model(x)
        # print(x.size(),'2')
        if self.downsample != None:
            x = self.downsample(x)
        # x = torch.cat((x, skip_input), 1)
        return x


class Generator_cnn_pathch4_844_48_3_nouplayer_server5(nn.Module):
    r"""cnn
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = my_cnn(
                in_size=int(embed_dim * 2**i_layer),
                out_size=int(embed_dim * 2**i_layer),
                dropout=0.0,
                downsample=(
                    PatchMerging_cnn
                    if (i_layer < self.num_layers - 1)
                    else None
                ),
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = my_cnn(
                    in_size=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer)
                    ),
                    out_size=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer)
                    ),
                    dropout=0.0,
                    downsample=Upsample_promotion_cnn,
                )
            else:
                layer = my_cnn(
                    in_size=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    out_size=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    dropout=0.0,
                    downsample=Upsample_cnn,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion_cnn(dim=embed_dim * 2),
            Upsample_promotion_cnn(dim=embed_dim),
            #
            # Upsample_layer(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, output_dim =embed_dim, norm_layer=norm_layer),
            # Upsample_layer(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                dim=embed_dim, output_dim=embed_dim // 2, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)

        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.shape)
        x = x.view(-1, 96, 64, 64)
        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        # print(x1.size(), 'x1')
        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            # print(x1.size(),'x1')
            if i < 3:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[1 - i]), 1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        # x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_8421_48_3_nouplayer_server5(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 2, 1],
        num_heads=[8, 8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        self.updepths = list(reversed(depths))
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            #
            # Upsample_layer(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, output_dim =embed_dim, norm_layer=norm_layer),
            # Upsample_layer(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                dim=embed_dim, output_dim=embed_dim // 2, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            # print(x1.size())
            if i < 4:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[2 - i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch4_844_48_3_residual(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            # nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size=3, padding=1, stride=1),
            self.resi_connection.append(layer)
        # self.resi_connection = nn.Sequential(
        #     nn.Conv2d(embed_dim *2, embed_dim *2, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1, stride=1),
        # )

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                output_dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
                norm_layer=norm_layer,
            ),
            # nn.Conv2d(embed_dim * 2 , embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim, norm_layer=norm_layer),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # Upsample_promotion(input_resolution=(patches_resolution[0]*2, patches_resolution[1]*2),
            #                                dim=embed_dim, norm_layer=norm_layer),
            # self.flinal_layer
            # Upsample_promotion(input_resolution=(patches_resolution[0], patches_resolution[1]),
            #                    dim=embed_dim*2, norm_layer=norm_layer),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            # nn.Linear(96, 30, bias=False),
            # nn.LeakyReLU(),
            # nn.Linear(48, 10, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.Linear(30, 3, bias=False),
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            # print(x.size())
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x
        # print('x1',x.size(),len(self.downsample_result))
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            if i < 3:
                x1 = torch.cat(
                    (
                        x1,
                        self.resi_connection[1 - i](
                            self.downsample_result[1 - i]
                        ),
                    ),
                    -1,
                )
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)

        # print(x.size())

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Generator_transformer_pathch2_no_Unt(nn.Module):
    r"""patch size is 2."""

    r""" Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 2
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=2,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[4, 6],
        num_heads=[3, 6],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer != -1:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )

            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim // 2), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            # nn.Linear(int(embed_dim //2), 3, bias=False),
            # nn.Conv2d(int(embed_dim/2), 3, 3, padding=1),
            # nn.Conv2d()
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Conv2d(int(embed_dim // 2), 3, 1, padding=0),
            nn.Tanh(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
            self.downsample_result.append(x)
        i = 0

        for xx in self.downsample_result:
            i = i + 1
        x1 = x

        i = 0
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            # if i < 1:
            #     x1 = torch.cat((x1, self.downsample_result[0-i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())
        x = self.final_upsample(x)

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.size())
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)

        # x = self.final(x)

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Discriminator_transformer(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=6,
        num_classes=1,
        embed_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)

        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x1, x2):
        # print(x1.size(),x2.size())
        x = torch.cat((x1, x2), 1)
        # print('forward',x.size())
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Discriminator_transformer2(nn.Module):
    ##add the postion.
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=6,
        num_classes=1,
        embed_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)

        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x1, x2):
        # print(x1.size(),x2.size())
        x = torch.cat((x1, x2), 1)
        # print('forward',x.size())
        x = self.forward_features(x)
        # print('forward', x.size()
        # x = x.permute(0, 2, 1)  # B C ,H*W
        # x = x.view([-1, 24, 256, 256])
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class Discriminator_transformer3(nn.Module):
    ##add the postion.patch gan.
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=6,
        num_classes=1,
        embed_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):

        x = self.patch_embed(x)

        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward(self, x1, x2):
        # print(x1.size(),x2.size())
        x = torch.cat((x1, x2), 1)
        # print('forward',x.size())

        x = self.forward_features(x)
        # print('forward', x.size())
        # x = x.permute(0, 2, 1)  # B C ,H*W
        # x = x.view([-1, 24, 256, 256])
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":
    test_data = np.load(
        "/home/devel/olimp/pyolimp/tests/test_data/test.npy", allow_pickle=True
    )

    test = test_data[3]
    test_t = torch.tensor(test).unsqueeze(0)

    swd_swin = Generator_transformer_pathch4_844_48_3_nouplayer_server5()
    state_dict = torch.load(
        "/home/devel/olimp/pyolimp/olimp/weights/cvd_swin.pth",
        map_location=torch.get_default_device(),
        weights_only=True,
    )
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    swd_swin.load_state_dict(new_state_dict)

    output = swd_swin(test_t)

    plt.imshow(output.detach().cpu().numpy().transpose([0, 2, 3, 1])[0])
    plt.savefig("fig1.png")
    plt.show()
