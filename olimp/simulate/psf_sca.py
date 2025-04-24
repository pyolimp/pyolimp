from __future__ import annotations
import torch
from math import pi, cos, sin


class PSFSCA:
    """
    Create optic PSF for fixed viewing distance 400 cm and canvas size 50 cm.
    Loosely based on https://github.com/scottmsul/EyeSimulator/blob/master/Geometric_Optics_2_Defocus.ipynb

    :returns: Tensor of size (height, width)
    """

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        x = torch.arange(width, dtype=torch.float32)
        y = torch.arange(height, dtype=torch.float32)
        x = x - width * 0.5 + 0.5  # +0.5 to be in the exact middle of the psf
        y = y - height * 0.5 + 0.5
        self._y, self._x = torch.meshgrid(y, x, indexing="ij")

    def __call__(
        self,
        sphere_dpt: float = -1.0,
        cylinder_dpt: float = 0.0,
        angle_rad: float = 0.0,
        pupil_diameter_mm: float = 4.0,
        am2px: float = 0.001,
    ) -> torch.Tensor:
        x, y = self._x, self._y
        blur_angle = pi * 0.5 + angle_rad

        # Radius of pupil
        r = pupil_diameter_mm * 0.5

        cos_theta = cos(blur_angle)
        sin_theta = sin(blur_angle)

        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y

        # Convert radius to angle minutes
        rad2am = 180.0 / pi * 60.0

        # Ellipse semi-axis
        a = r * abs(sphere_dpt + cylinder_dpt) * rad2am * am2px
        b = r * abs(sphere_dpt) * rad2am * am2px

        # Distance map
        dist = torch.square(x_rot / a) + torch.square(y_rot / b)

        # Setting an ellipse
        kernel = (dist <= 1).float()

        # Normalizing
        kernel /= kernel.sum()

        return kernel
