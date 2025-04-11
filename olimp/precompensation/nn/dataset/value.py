from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
import torch
from ballfish import DistributionParams, create_distribution


class ValueDataset(Dataset[Tensor]):
    """
    Treat distribution as a dataset of scalars. Currently used to augment
    `hue_angle_deg` in `ColorBlindnessDistortion`
    """

    def __init__(
        self,
        value: DistributionParams,
        seed: int = 42,
        size: int = 10000,
    ):
        self._seed = seed
        self._size = size
        self._value = create_distribution(value)

    def __getitem__(self, index: int) -> Tensor:
        random = Random(f"{self._seed}|{index}")
        return torch.tensor(self._value(random))

    def __len__(self):
        return self._size
