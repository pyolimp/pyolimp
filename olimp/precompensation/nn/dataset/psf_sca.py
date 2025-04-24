from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
from ballfish import DistributionParams, create_distribution
from math import radians
from olimp.simulate.psf_sca import PSFSCA


class PSFSCADataset(Dataset[Tensor]):
    def __init__(
        self,
        width: int,
        height: int,
        sphere_dpt: DistributionParams = -1,
        cylinder_dpt: DistributionParams = 0.0,
        angle_deg: DistributionParams = 0.0,
        pupil_diameter_mm: DistributionParams = 4.0,
        am2px: float = 0.001,
        seed: int = 42,
        size: int = 10000,
    ):
        self._seed = seed
        self._size = size
        self._sphere_dpt = create_distribution(sphere_dpt)
        self._cylinder_dpt = create_distribution(cylinder_dpt)
        self._angle_deg = create_distribution(angle_deg)
        self._pupil_diameter_mm = create_distribution(pupil_diameter_mm)
        self._am2px = am2px
        self._generator = PSFSCA(width=width, height=height)

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

    def __len__(self):
        return self._size
