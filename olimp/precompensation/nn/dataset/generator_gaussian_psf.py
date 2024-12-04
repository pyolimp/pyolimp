from __future__ import annotations
from typing import Callable, Generic, TypeVar, TypedDict
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class NormParams(TypedDict):
    center: int
    sigma: float
    a: float
    b: float


class KernelGaussianDataset(Dataset[Tensor]):
    def __init__(self, shape: tuple[int, int], x: NormParams, y: NormParams, theta: NormParams, sigma_x: NormParams, sigma_y: NormParams):
      self.shape = shape
      self.theta = theta # make assert
      self.center = (x['center'], y['center'])
      self.sigma_x = x['sigma']
      self.sigma_y = y['sigma']


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
        height, width = self.shape
        y_center, x_center = self.center

        # Create grid of (x, y) coordinatesТы завтра пойдёшь сдавать?
        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing="ij")

        # Shift coordinates to the center
        x_shifted = x - x_center
        y_shifted = y - y_center

        # Apply rotation
        cos_theta = torch.cos(torch.tensor(self.theta))
        sin_theta = torch.sin(torch.tensor(self.theta))
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        gaussian = torch.exp(-(x_rot**2 / (2 * self.sigma_x**2) + y_rot**2 / (2 * self.sigma_y**2)))

        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian


    def __len__(self):
        return 1000
