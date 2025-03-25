import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from ..cs.srgb import sRGB
from ..cs.oklab import Oklab
from ..cs.opponent import Opponent
from olimp.processing import fft_conv


def _srgb2opponent(srgb: Tensor) -> Tensor:
    return Opponent().from_XYZ(sRGB().to_XYZ(srgb))


def _opponent2oklab(oppo: Tensor) -> Tensor:
    return Oklab().from_XYZ(Opponent().to_XYZ(oppo).clip(min=0.0))


def _create_gauss_kernel_2d(
    sigma: Tensor, weight: Tensor, height: int, width: int
) -> Tensor:
    # https://en.wikipedia.org/wiki/Gaussian_blur
    y = (torch.arange(height, dtype=torch.float32) - height * 0.5)[:, None]
    x = (torch.arange(width, dtype=torch.float32) - width * 0.5)[:, None].T
    kernel = torch.zeros(height, width)
    for s, w in zip(sigma, weight):
        gauss = torch.exp(-(x * x + y * y) / (2 * s * s))
        kernel += gauss / gauss.sum() * w
    return torch.fft.fftshift(kernel / kernel.sum())


def _image_metric(
    A_srgb: Tensor,
    B_srgb: Tensor,
    spacial_filters: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    assert A_srgb.shape == B_srgb.shape, (A_srgb.shape, B_srgb.shape)
    assert A_srgb.ndim == 3, A_srgb.ndim
    assert len(spacial_filters) == 3

    A = _srgb2opponent(A_srgb)
    B = _srgb2opponent(B_srgb)

    A_convolved = torch.zeros_like(A)
    B_convolved = torch.zeros_like(A)
    for img_src, img_dst in ((A, A_convolved), (B, B_convolved)):
        for ch_idx, kernel in enumerate(spacial_filters):
            img_dst[ch_idx] = fft_conv(img_src[ch_idx], kernel)
    A_metric_cs = _opponent2oklab(A_convolved)
    B_metric_cs = _opponent2oklab(B_convolved)

    metric = torch.linalg.norm(A_metric_cs - B_metric_cs, axis=0)

    return torch.mean(metric)


class SOkLab(Module):
    """
    Code is based on:
    https://github.com/iitpvisionlab/vsl_ial/blob/main/vsl_ial/image_metric.py
    """

    def __init__(self, dpi: float, distance_inch: float):

        ppd = dpi * distance_inch * torch.tan(torch.tensor(torch.pi / 180))

        self.weights = (
            torch.tensor((1.00327, 0.11442, -0.11769)),
            torch.tensor((0.61672, 0.38328)),
            torch.tensor((0.56789, 0.43211)),
        )
        self.sigmas = (
            torch.tensor((0.0283, 0.133, 4.336)) * ppd,
            torch.tensor((0.0392, 0.494)) * ppd,
            torch.tensor((0.0536, 0.386)) * ppd,
        )

        super().__init__()

    def forward(self, image1: Tensor, image2: Tensor):
        assert image1.ndim == 4, image1.shape
        assert image2.ndim == 4, image2.shape

        assert image1.shape[1] == 3
        assert image2.shape[1] == 3
        h, w = image1.shape[-2:]

        s_oklab_values = torch.empty((image1.shape[0]))

        spacial_filters = (
            _create_gauss_kernel_2d(self.sigmas[0], self.weights[0], h, w),
            _create_gauss_kernel_2d(self.sigmas[1], self.weights[1], h, w),
            _create_gauss_kernel_2d(self.sigmas[2], self.weights[2], h, w),
        )

        for idx in range(image1.shape[0]):
            s_oklab_values[idx] = _image_metric(
                image1[idx], image2[idx], spacial_filters
            )
        return s_oklab_values
