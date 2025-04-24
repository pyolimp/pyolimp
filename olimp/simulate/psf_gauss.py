from __future__ import annotations
import torch
from math import cos, sin


class PSFGauss:
    """
    Class for generating Gaussian PSF

    :returns: Tensor of size (height, width)
    """

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        x = torch.arange(width, dtype=torch.float32) + 0.5
        y = torch.arange(height, dtype=torch.float32) + 0.5
        self._y, self._x = torch.meshgrid(y, x, indexing="ij")

    def __call__(
        self,
        center_x: float,
        center_y: float,
        theta: float,
        sigma_x: float,
        sigma_y: float,
    ) -> torch.Tensor:
        assert sigma_x > 1e-6 and sigma_y > 1e-6
        # Shift coordinates to the center
        x_shifted = self._x - center_x
        y_shifted = self._y - center_y

        # Apply rotation
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Compute the Gaussian function
        gaussian = torch.exp(
            -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
        )
        # Normalize the Gaussian to have a maximum value of 1
        gaussian /= gaussian.sum()

        return gaussian
