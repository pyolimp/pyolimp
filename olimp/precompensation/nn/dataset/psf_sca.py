from __future__ import annotations
from random import Random

import torch
from torch import Tensor
from typing import Callable, Generator
from ballfish import DistributionParams, create_distribution
from math import radians
from olimp.simulate.psf_sca import PSFSCA
from olimp.simulate.refraction_distortion import RefractionDistortion
from olimp.precompensation.nn.dataset.distortion_dataset import (
    DistortionDataset,
)


class PSFSCADataset(DistortionDataset):
    def __init__(
        self,
        width: int,
        height: int,
        sphere_dpt: DistributionParams = -1.0,
        cylinder_dpt: DistributionParams = 0.0,
        angle_deg: DistributionParams = 0.0,
        pupil_diameter_mm: DistributionParams = 4.0,
        am2px: float = 0.001,
        seed: int = 42,
        size: int = 10000,
    ):
        super().__init__(
            seed, size, generator=PSFSCA(width=width, height=height)
        )
        self._sphere_dpt = create_distribution(sphere_dpt)
        self._cylinder_dpt = create_distribution(cylinder_dpt)
        self._angle_deg = create_distribution(angle_deg)
        self._pupil_diameter_mm = create_distribution(pupil_diameter_mm)
        self._am2px = am2px
        self.refraction_distorion = RefractionDistortion()

    def __getitem__(self, index: int) -> Tensor:
        random = Random(f"{self._seed}|{index}")
        psf = self._generator(
            sphere_dpt=self._sphere_dpt(random),
            cylinder_dpt=self._cylinder_dpt(random),
            angle_rad=radians(self._angle_deg(random)),
            pupil_diameter_mm=self._pupil_diameter_mm(random),
            am2px=self._am2px,
        )
        return psf[None]

    def apply(self) -> Callable[[Tensor], Generator[Tensor, None, None]]:
        def _apply(image: Tensor) -> Generator[Tensor, None, None]:
            for index in range(self._size):
                psf = self.__getitem__(index).to(image.device)
                psf = torch.fft.fftshift(psf)
                yield self.refraction_distorion(psf)(image)

        return _apply
