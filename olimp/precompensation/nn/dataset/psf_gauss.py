from __future__ import annotations
from random import Random

import torch
from torch import Tensor
from typing import Callable, Generator
from ballfish import DistributionParams, create_distribution
from math import radians
from olimp.simulate.psf_gauss import PSFGauss
from olimp.precompensation.nn.dataset.distortion_dataset import (
    DistortionDataset,
)
from olimp.simulate.refraction_distortion import RefractionDistortion


class PsfGaussDataset(DistortionDataset):
    def __init__(
        self,
        width: int,
        height: int,
        center_x: DistributionParams,
        center_y: DistributionParams,
        theta: DistributionParams,
        sigma_x: DistributionParams,
        sigma_y: DistributionParams,
        seed: int = 42,
        size: int = 10000,
    ):
        super().__init__(
            seed, size, generator=PSFGauss(width=width, height=height)
        )
        self._theta = create_distribution(theta)
        self._center_x = create_distribution(center_x)
        self._center_y = create_distribution(center_y)
        self._sigma_x = create_distribution(sigma_x)
        self._sigma_y = create_distribution(sigma_y)
        self.refraction_distorion = RefractionDistortion()

    def __getitem__(self, index: int) -> Tensor:
        random = Random(f"{self._seed}|{index}")
        gaussian = self._generator(
            center_x=self._center_x(random),
            center_y=self._center_y(random),
            theta=radians(self._theta(random)),
            sigma_x=self._sigma_x(random),
            sigma_y=self._sigma_y(random),
        )
        return gaussian[None]

    def apply(self) -> Callable[[Tensor], Generator[Tensor, None, None]]:
        def _apply(image: Tensor) -> Generator[Tensor, None, None]:
            for index in range(self._size):
                psf = self.__getitem__(index).to(image.device)
                psf = torch.fft.fftshift(psf)
                yield self.refraction_distorion(psf)(image)

        return _apply
