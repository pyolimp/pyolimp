import torch
from torch import Tensor

from olimp.simulate import Distortion
from olimp.processing import fft_conv


class RefractionDistortion(Distortion):
    def __init__(self, psf: Tensor) -> None:
        assert psf.dtype == torch.float32, psf.dtype
        self.psf = psf

    def __call__(self, image: Tensor) -> Tensor:
        return fft_conv(image, self.psf)
