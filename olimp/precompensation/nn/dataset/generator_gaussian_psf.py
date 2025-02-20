from __future__ import annotations
from typing import Tuple, Dict
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
import torch
from ballfish.distribution import DistributionParams, create_distribution
import math

import matplotlib.pyplot as plt


class KernelGaussianDataset(Dataset[Tensor]):
    name = "kernel_distribution"
    default_distribution: Dict[str, DistributionParams] = {
        "center": {
            "name": "truncnorm",
            "a": 256,
            "b": 256,
            "mu": 256,
            "sigma": 2,
        },
        "theta": {
            "name": "truncnorm",
            "a": -3 * math.pi,
            "b": 3 * math.pi,
            "mu": 0,
            "sigma": 10,
        },
        "sigma": {
            "name": "truncnorm",
            "a": 10,
            "b": 20,
            "mu": 15,
            "sigma": 30.25,
        },
    }

    def __init__(
        self,
        shape: Tuple[int, int],
        random: Random = Random(42),
        center: Tuple[DistributionParams, DistributionParams] = (
            default_distribution["center"],
            default_distribution["center"],
        ),
        theta: DistributionParams = default_distribution["theta"],
        sigma_x: DistributionParams = default_distribution["sigma"],
        sigma_y: DistributionParams = default_distribution["sigma"],
    ):
        self.shape = shape
        self.random = random
        self.theta = create_distribution(theta)
        self.center = (
            create_distribution(center[0]),
            create_distribution(center[1]),
        )
        self.sigma_x = create_distribution(sigma_x)
        self.sigma_y = create_distribution(sigma_y)

    @staticmethod
    def guassian_generator(
        shape: Tuple[int, int],
        random: Random = Random(42),
        center: Tuple[DistributionParams, DistributionParams] = (
            default_distribution["center"],
            default_distribution["center"],
        ),
        theta: DistributionParams = default_distribution["theta"],
        sigma_x: DistributionParams = default_distribution["sigma"],
        sigma_y: DistributionParams = default_distribution["sigma"],
    ) -> Tensor:
        width, height = shape
        theta_distr = create_distribution(theta)
        center_distr = (
            create_distribution(center[0]),
            create_distribution(center[1]),
        )
        sigma_x_distr = create_distribution(sigma_x)
        sigma_y_distr = create_distribution(sigma_y)

        # Create grid of (x, y) coordinates
        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing="ij")

        # Shift coordinates to the center
        center = (center_distr[0](random), center_distr[1](random))
        x_center, y_center = center
        x_shifted = x - x_center
        y_shifted = y - y_center

        # Apply rotation
        theta = theta_distr(random)
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        sigma_x = sigma_x_distr(random)
        sigma_y = sigma_y_distr(random)
        gaussian = torch.exp(
            -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
        )
        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian

    def __getitem__(self, index: int) -> Tensor:
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

        width, height = self.shape
        # Create grid of (x, y) coordinates
        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing="ij")

        # Shift coordinates to the center
        center = (self.center[0](self.random), self.center[1](self.random))
        x_center, y_center = center
        x_shifted = x - x_center
        y_shifted = y - y_center

        # Apply rotation
        theta = self.theta(self.random)
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        sigma_x = self.sigma_x(self.random)
        sigma_y = self.sigma_y(self.random)
        gaussian = torch.exp(
            -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
        )
        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian

    def __len__(self):
        return 1000


if __name__ == "__main__":
    object = KernelGaussianDataset((512, 512))
    for i in range(1):
        tensor = object[i]
        plt.figure(figsize=(10, 10))
        plt.imshow(tensor.cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title("Tensor Visualization")
        plt.show()
