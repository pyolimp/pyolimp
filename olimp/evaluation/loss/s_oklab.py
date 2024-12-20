from math import ceil

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from ..cs.srgb import sRGB
from ..cs.oklab import Oklab
from ..cs.opponent import Opponent


def srgb2opponent(srgb: Tensor) -> Tensor:
    srgb = sRGB().to_XYZ(srgb)
    return Opponent().from_XYZ(srgb)


def opponent2oklab(oppo: Tensor) -> Tensor:
    return Oklab().from_XYZ(Opponent().to_XYZ(oppo))


def create_gauss_kernel_2d(sigma: float, weight: float = 1.0):
    # https://en.wikipedia.org/wiki/Gaussian_blur
    size = ceil(sigma * 3.5)
    x = torch.arange(-size, size + 1)[:, None]
    y = x.T
    gauss = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss *= weight / gauss.sum()
    return gauss


def image_metric(
    A_srgb: Tensor,
    B_srgb: Tensor,
    spacial_filters: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    assert A_srgb.shape == B_srgb.shape, (A_srgb.shape, B_srgb.shape)
    assert A_srgb.ndim == 3, A_srgb.ndim
    assert len(spacial_filters) == 3
    for filter in spacial_filters:
        for sz in filter.shape:
            assert sz % 2 == 1, f"filter size must be odd (not {sz})"

    A = srgb2opponent(A_srgb)
    B = srgb2opponent(B_srgb)

    A_convolved = torch.zeros_like(A)
    B_convolved = torch.zeros_like(A)
    for img_src, img_dst in ((A, A_convolved), (B, B_convolved)):
        for ch_idx, kernel in enumerate(spacial_filters):
            w, h = kernel.shape[-2:]
            assert w == h
            sz = w // 2  # can be easily changed to support non square kernels
            pad = sz
            img_dst[:, :, ch_idx] = F.conv2d(
                img_src[:, :, ch_idx][None, None],
                kernel[None, None],
                padding=pad,
            )
    A_metric_cs = opponent2oklab(A_convolved)
    B_metric_cs = opponent2oklab(B_convolved)
    metric = torch.linalg.norm(A_metric_cs - B_metric_cs, axis=2)

    return torch.mean(metric)


class SOkLab(Module):

    def __init__(self):
        super().__init__()

    def forward(self, img1: Tensor, img2: Tensor):
        assert img1.ndim == 4, img1.shape
        assert img2.ndim == 4, img2.shape

        assert img1.shape[1] == 3
        assert img2.shape[1] == 3
        s_oklab_values = torch.empty((img1.shape[0]))
        for idx in range(img1.shape[0]):
            s_oklab_values[idx] = image_metric(
                img1[idx].permute(1, 2, 0),
                img2[idx].permute(1, 2, 0),
                (
                    create_gauss_kernel_2d(1.0),
                    create_gauss_kernel_2d(1.0),
                    create_gauss_kernel_2d(1.0),
                ),
            )
        return s_oklab_values
