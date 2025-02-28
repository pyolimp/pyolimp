from __future__ import annotations
from torch import Tensor
from olimp.simulate import ApplyDistortion
from olimp.processing import fft_conv


class RefractionDistortion:
    @staticmethod
    def __call__(psf: Tensor) -> ApplyDistortion:
        return lambda image: fft_conv(image, psf)
