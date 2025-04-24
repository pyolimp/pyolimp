from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
from ballfish import DistributionParams, create_distribution
from math import radians
from olimp.simulate.psf_gauss import PSFGauss


class PsfGaussDataset(Dataset[Tensor]):
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
        self._seed = seed
        self._size = size
        self._theta = create_distribution(theta)
        self._center_x = create_distribution(center_x)
        self._center_y = create_distribution(center_y)
        self._sigma_x = create_distribution(sigma_x)
        self._sigma_y = create_distribution(sigma_y)
        self._generator = PSFGauss(width=width, height=height)

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

    def __len__(self):
        return self._size
