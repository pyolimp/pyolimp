from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from olimp.processing import fft_conv, scale_value


def feng_xu(image: Tensor, psf: Tensor, lambda_val: float = 7.0) -> Tensor:
    """
    .. image:: ../../_static/feng_xu.svg
       :class: full-width
    """
    f1 = torch.tensor([[[[1, -1]]]])
    f2 = torch.tensor([[[[1], [-1]]]])

    # Compute Fourier transforms
    F_Io = torch.fft.fft2(image)
    F_K = torch.fft.fft2(psf)
    F_f1 = torch.fft.fft2(f1, dim=(0, 1, 2, 3), s=image.shape)
    F_f2 = torch.fft.fft2(f2, dim=(0, 1, 2, 3), s=image.shape)

    # Compute conjugates and element-wise multiplication
    F_K_star = torch.conj(F_K)
    F_f1_star = torch.conj(F_f1)
    F_f2_star = torch.conj(F_f2)

    numerator = lambda_val * (F_K_star * F_Io)
    denominator = (
        lambda_val * (F_K_star * F_K) + (F_f1_star * F_f1) + (F_f2_star * F_f2)
    )

    # Compute the pre-corrected image
    F_Ip = numerator / denominator
    precompensation = torch.real(torch.fft.ifft2(F_Ip))
    return precompensation


def _demo():
    from .._demo import demo

    def demo_huang(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        ret = feng_xu(image, psf, lambda_val=2)
        progress(0.8)
        ret = fft_conv(scale_value(ret), psf)
        progress(1.0)
        return ret

    demo("Feng Xu", demo_huang, mono=True)


if __name__ == "__main__":
    _demo()
