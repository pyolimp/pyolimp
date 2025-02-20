from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
import torch
from ballfish.distribution import DistributionParams, create_distribution


class KernelGaussianDataset(Dataset[Tensor]):
    name = "kernel_distribution"

    def __init__(
        self,
        width: int,
        height: int,
        center_x: DistributionParams,
        center_y: DistributionParams,
        theta: DistributionParams,
        sigma_x: DistributionParams,
        sigma_y: DistributionParams,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.random = Random(seed)
        self.theta = create_distribution(theta)
        self.center_x = create_distribution(center_x)
        self.center_y = create_distribution(center_y)
        self.sigma_x = create_distribution(sigma_x)
        self.sigma_y = create_distribution(sigma_y)

    @staticmethod
    def guassian_generator(
        width: int,
        height: int,
        center_x: float,
        center_y: float,
        theta: float,
        sigma_x: float,
        sigma_y: float,
    ) -> Tensor:
        """
        Generate a 2D elliptical Gaussian distribution over a specified shape.

        Parameters:
        - shape: tuple of ints, (height, width) of the output array.
        - center: tuple of floats, (y_center, x_center) of the ellipse center.
        - sigma_x: float, standard deviation along the x-axis.
        - sigma_y: float, standard deviation along the y-axis.
        - theta: float, rotation angle in radians.

        Returns:
        - gaussian: 2D Tensor representing the elliptical Gaussian distribution.
        """

        # Create grid of (x, y) coordinates
        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing="ij")

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
        gaussian = self.guassian_generator(
            width=self.width,
            height=self.height,
            center_x=self.center_x(self.random),
            center_y=self.center_y(self.random),
            theta=self.theta(self.random),
            sigma_x=self.sigma_x(self.random),
            sigma_y=self.sigma_y(self.random),
        )
        return gaussian

    def __len__(self):
        return 10000
