from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from olimp.processing import scale_value


def huang(image: Tensor, psf: Tensor, k: float = 0.01) -> Tensor:
    """
    Inverse blur filtering. Single channel only.
    """
    assert (
        image.shape[-2:] == psf.shape[-2:]
    ), f"Expected equal shapes, got: image={image.shape}, psf={psf.shape}"

    k = min(max(1e-7, k), 0.025)

    otf = torch.fft.fftn(psf, dim=(-2, -1))
    mtf = torch.abs(otf)

    # normalize MTF & OTF
    mtf_max = mtf.max()  # change to mtf[..., 0, 0] once tested properly
    mtf = mtf / mtf_max
    otf = otf / mtf_max

    # we manually set values for ratio in IBF as {num} / {denum}
    # to be able to handle zero division cases
    num = torch.fft.fftn(image, dim=(-2, -1)) * torch.square(mtf)
    denum = otf * (torch.square(mtf) + k)

    ratio = torch.zeros_like(num)
    mask = denum != 0
    ratio[mask] = torch.div(num[mask], denum[mask])

    # the result is already shifted to center
    result = torch.fft.ifftn(ratio, dim=(-2, -1), norm="backward")
    return torch.real(result)


def _demo():
    from .._demo import demo

    def demo_huang(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        ret = huang(image, psf)
        progress(1.0)
        return scale_value(ret, min_val=0, max_val=1.0)

    demo("Huang", demo_huang, mono=True)


if __name__ == "__main__":
    _demo()
