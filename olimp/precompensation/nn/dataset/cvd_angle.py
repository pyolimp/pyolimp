from __future__ import annotations
from random import Random
from typing import Callable, Generator

import torch
from torch import Tensor
from ballfish import DistributionParams, create_distribution

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
from olimp.precompensation.nn.dataset.distortion_dataset import (
    DistortionDataset,
)


class ColorBlindnessDataset(DistortionDataset):
    def __init__(
        self,
        angle_deg: DistributionParams,
        seed: int = 42,
        size: int = 365,
    ) -> None:
        super().__init__(seed, size, generator=None)
        self._angle = create_distribution(angle_deg)
        self._angles: list[float] = []
        self._distortions: list[ColorBlindnessDistortion] = []

        for i in range(size):
            random = Random(f"{seed}|{i}")
            angle = self._angle(random)
            self._angles.append(angle)
            self._distortions.append(ColorBlindnessDistortion(angle))

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor(self._angles[index])

    def apply(self) -> Callable[[Tensor], Generator[Tensor, None, None]]:
        def _apply(image: Tensor) -> Generator[Tensor, None, None]:
            for distortion in self._distortions:
                yield distortion()(image)

        return _apply
