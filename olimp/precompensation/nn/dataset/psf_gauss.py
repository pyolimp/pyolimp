from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
import torch
from ballfish.distribution import DistributionParams, create_distribution
from math import radians


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
        self._width = width
        self._height = height
        self._seed = seed
        self._size = size
        self._theta = create_distribution(theta)
        self._center_x = create_distribution(center_x)
        self._center_y = create_distribution(center_y)
        self._sigma_x = create_distribution(sigma_x)
        self._sigma_y = create_distribution(sigma_y)

        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        self._y, self._x = torch.meshgrid(y, x, indexing="ij")

    @staticmethod
    def guassian_generator(
        x: Tensor,
        y: Tensor,
        center_x: float,
        center_y: float,
        theta: float,
        sigma_x: float,
        sigma_y: float,
    ) -> Tensor:
        """
        Generate a 2D elliptical Gaussian distribution over a specified shape.

        Parameters:
        - x, y: meshgrid
        - center_x, center_y: ellipse center.
        - sigma_x, sigma_y: float, standard deviation along the x/y-axis.
        - theta: float, rotation angle in radians.

        Returns:
        - gaussian: 2D Tensor representing the elliptical Gaussian distribution.
        """

        # Shift coordinates to the center
        x_shifted = x - center_x
        y_shifted = y - center_y

        # Apply rotation
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        gaussian = torch.exp(
            -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
        )
        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian

    def __getitem__(self, index: int) -> Tensor:
        random = Random(f"{self._seed}|{index}")
        gaussian = self.guassian_generator(
            x=self._x,
            y=self._y,
            center_x=self._center_x(random),
            center_y=self._center_y(random),
            theta=radians(self._theta(random)),
            sigma_x=self._sigma_x(random),
            sigma_y=self._sigma_y(random),
        )
        return gaussian[None]

    def __len__(self):
        return self._size
