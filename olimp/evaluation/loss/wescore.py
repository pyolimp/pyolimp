from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab as ProLabCS

# from ._base import ReducibleLoss, Reduction


def extract_mixed_patches(
    img: Tensor, kernel_size: int, stride: int
) -> list[Tensor]:

    B, C, H, W = img.shape
    patches = []

    full_h = ((H + stride - kernel_size) // stride - 1) * stride + kernel_size
    full_w = ((H + stride - kernel_size) // stride - 1) * stride + kernel_size

    # 1. main windows, overlapping, without edges (hxw)
    main = img[:, :, :full_h, :full_w]
    main_patches = main.unfold(2, kernel_size, stride).unfold(
        3, kernel_size, stride
    )
    # (B, C, H_k, W_k, h, w) → (B, H_k, W_k, C, h, w)
    main_patches = main_patches.permute(0, 2, 3, 1, 4, 5)
    patches.append(main_patches)

    return patches  # (B, H_k, W_k, C, h, w)


def _get_indices(tensor: Tensor, mul: int = 5) -> tuple[Tensor, Tensor]:
    _, window_h, window_w, _, height_pixel, width_pixel = tensor.shape
    num = window_h * window_w
    length = height_pixel * width_pixel
    total_length = length * mul

    torch.manual_seed(0)

    random_values = torch.rand(num * mul, length)
    t1 = torch.argsort(random_values, dim=1)
    random_values = torch.rand(num * mul, length)
    t2 = torch.argsort(random_values, dim=1)

    t1 = t1.reshape(num, total_length)
    t2 = t2.reshape(num, total_length)

    return t1, t2


# def CCPR_torch(img0: Tensor, img1: Tensor, thr: int) -> Tensor:
#     img0_mask = img0 > thr
#     den = torch.clamp(img0_mask.sum(dim=(1, 2)), min=1)

#     img1_mask = (img1 * img0_mask) > thr
#     ccpr = img1_mask.sum(dim=(1, 2)) / den
#     return ccpr

# def CCFR_torch(img1: Tensor, img0: Tensor, thr: int) -> Tensor:
#     img1_mask = img1 > thr
#     den = torch.clamp(img1_mask.sum(dim=(1, 2)), min=1)

#     img0_mask = (img1_mask == 1) & (img0 < thr)
#     num = img0_mask.sum(dim=(1, 2))
#     ccfr = 1 - num / den
#     return ccfr


def _ccpr_torch(
    img0: Tensor, img1: Tensor, thr: float, eps: float = 1e-6
) -> Tensor:  # not precise, changed to make differentiable
    img0_mask = torch.clamp(img0 - thr + 0.5, 0.0, 1.0)
    den = img0_mask.sum(dim=(1, 2)) + eps
    img1_mask = torch.clamp((img1 * img0_mask) - thr + 0.5, 0.0, 1.0)

    num = img1_mask.sum(dim=(1, 2)) + eps
    ccpr = num / den
    return ccpr


def _ccfr_torch(
    img1: Tensor, img0: Tensor, thr: float, eps: float = 1e-6
) -> Tensor:  # not precise, changed to make differentiable
    img1_mask = torch.clamp(img1 - thr + 0.5, 0.0, 1.0)
    den = img1_mask.sum(dim=(1, 2)) + eps
    img0_inv_mask = torch.clamp(0.5 - (img0 - thr), 0.0, 1.0)

    num = (img1_mask * img0_inv_mask).sum(dim=(1, 2))
    ccfr = 1.0 - num / den
    return ccfr


def _get_init_torch(img: Tensor, t1: Tensor, t2: Tensor) -> Tensor:
    t1 = t1.unsqueeze(0).unsqueeze(2).expand(-1, -1, 3, -1)
    t2 = t2.unsqueeze(0).unsqueeze(2).expand(-1, -1, 3, -1)
    wide = torch.gather(img, 3, t1) - torch.gather(img, 3, t2)
    color_contr = torch.abs(wide).sum(dim=2)
    return color_contr


def wescore_torch(
    img: Tensor,
    sim: Tensor,
    color_space: Literal["lab", "prolab"],
    lp: int,
    lf: int,
    thr: int,
    mull: int,
) -> Tensor:

    # img_ten = torch.unsqueeze(img, 0)
    # sim_ten = torch.unsqueeze(sim, 0)

    assert lp % 2 == 1 or lf % 2 == 1, "lp and lf should be odd"

    img_ten = img.clamp(0, 1)
    sim_ten = sim.clamp(0, 1)

    sim_list = []
    img_list = []
    if color_space == "lab":
        for i in range(sim_ten.shape[0]):
            sim_lab = (
                CIELAB(D65_sRGB)
                .from_XYZ(sRGB().to_XYZ(img_ten[i]))
                .unsqueeze(0)
            )
            img_lab = (
                CIELAB(D65_sRGB)
                .from_XYZ(sRGB().to_XYZ(sim_ten[i]))
                .unsqueeze(0)
            )
            sim_list.append(sim_lab)
            img_list.append(img_lab)

    elif color_space == "prolab":
        for i in range(sim_ten.shape[0]):
            sim_lab = (
                ProLabCS(D65_sRGB)
                .from_XYZ(sRGB().to_XYZ(img_ten[i]))
                .unsqueeze(0)
            )
            img_lab = (
                ProLabCS(D65_sRGB)
                .from_XYZ(sRGB().to_XYZ(sim_ten[i]))
                .unsqueeze(0)
            )
            sim_list.append(sim_lab)
            img_list.append(img_lab)

    sim_test = torch.cat(sim_list, dim=0).requires_grad_()
    img_test = torch.cat(img_list, dim=0).requires_grad_()

    image = img_test * 100
    simulation = sim_test * 100

    # del sim_ten, sim_list, sim_test, img_ten, img_list, img_test

    # CCPR
    stride_p = int(lp / 2)
    patches_img = extract_mixed_patches(image, lp, stride_p)[
        0
    ]  # Batch, window index height, wndow index width, Color, hei, wid
    patches_sim = extract_mixed_patches(simulation, lp, stride_p)[
        0
    ]  # Batch, window index height, wndow index width, Color, hei, wid

    t1, t2 = _get_indices(patches_img, mull)
    patches_img = patches_img.flatten(-2, -1)
    patches_sim = patches_sim.flatten(-2, -1)
    patches_img = patches_img.flatten(1, 2)
    patches_sim = patches_sim.flatten(1, 2)

    ccpr_o = _get_init_torch(patches_img, t1, t2)
    ccpr_s = _get_init_torch(patches_sim, t1, t2)

    ccpr = _ccpr_torch(ccpr_o, ccpr_s, thr)

    del lp, stride_p

    # CCFR
    stride_f = int(lf / 2)
    patches_img = extract_mixed_patches(image, lf, stride_f)[
        0
    ]  # Batch, window index height, wndow index width, Color, hei, wid
    patches_sim = extract_mixed_patches(simulation, lf, stride_f)[
        0
    ]  # Batch, window index height, wndow index width, Color, hei, wid

    t1, t2 = _get_indices(patches_img, mull)
    patches_img = patches_img.flatten(-2, -1)
    patches_sim = patches_sim.flatten(-2, -1)
    patches_img = patches_img.flatten(1, 2)
    patches_sim = patches_sim.flatten(1, 2)

    ccfr_o = _get_init_torch(patches_img, t1, t2)
    ccfr_s = _get_init_torch(patches_sim, t1, t2)

    ccfr = _ccfr_torch(ccfr_o, ccfr_s, thr)

    eps = 1e-8
    wE_score = 2 * ccpr * ccfr / (ccpr + ccfr + eps)
    wE_score = torch.clamp(wE_score, 0.0, 1.0)

    return 1.0 - wE_score


class WEscore(Module):
    _color_space: Literal["lab", "prolab"]

    def __init__(
        self,
        color_space: Literal["lab", "prolab"],
        window_size_p: int = 61,
        window_size_f: int = 7,
        threshold: int = 6,
        pair_multiplier: int = 5,
    ) -> None:
        super().__init__()
        self._color_space = color_space
        self._window_size_p = window_size_p
        self._window_size_f = window_size_f
        self._threshold = threshold
        self._pair_multiplier = pair_multiplier

    def forward(self, img1: Tensor, img2: Tensor):
        assert img1.ndim == 4, img1.shape
        assert img2.ndim == 4, img2.shape

        assert img1.shape[1] == 3
        assert img2.shape[1] == 3
        return wescore_torch(
            img1,
            img2,
            self._color_space,
            self._window_size_p,
            self._window_size_f,
            self._threshold,
            self._pair_multiplier,
        )
