from __future__ import annotations
from typing import Callable, Generic, TypeVar, TypedDict
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class NormParams(TypedDict):
    center: float
    sigma: float
    delta: float
    a: float
    b: float

class KernelGaussianDataset(Dataset[Tensor]):
    def __init__(self, shape: tuple[int, int], x: NormParams, y: NormParams, theta: NormParams, sigma_x: NormParams, sigma_y: NormParams):
      assert -2 * np.pi <= theta['center'] <= 2 * np.pi, f'theta={theta} is not in [-2π, 2π] range'
      assert sigma_x['center'] ** 2 > 0 and sigma_y['center'] ** 2 > 0, f'zero values in sigma requires'
      self.shape = shape
      self.theta = theta # make assert
      self.center = (x, y)
      self.sigma_x = sigma_x
      self.sigma_y = sigma_y
    
    def _generate_value(self, param: NormParams) -> float:
        if 'sigma' in param.keys():
          value = float(torch.normal(mean=param['center'], std=param['sigma'], size=(1,))[0])
        else:
          value = float(torch.normal(mean=float(param['center']), std=float(0), size=(1,))[0])
        
        if 'delta' in param.keys():
           lower = param['center'] - param['delta']
           upper = param['center'] + param['delta']
           value = float(torch.clamp(value, lower, upper))
        else:
          if 'a' in param.keys() and value < param['a']:
            value = param['a']
          if 'b' in param.keys() and value > param['b']:
            value = param['b']
        return value


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
        y_center, x_center = self._generate_value(self.center[0]), self._generate_value(self.center[1])
        sigma_x, sigma_y = self._generate_value(self.sigma_x), self._generate_value(self.sigma_y)
        theta = self._generate_value(self.theta)

        # Create grid of (x, y) coordinates
        y = torch.arange(height, dtype=torch.float32)
        x = torch.arange(width, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing="ij")

        # Shift coordinates to the center
        x_shifted = x - x_center
        y_shifted = y - y_center

        # Apply rotation
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        gaussian = torch.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian


    def __len__(self):
        return 1000
