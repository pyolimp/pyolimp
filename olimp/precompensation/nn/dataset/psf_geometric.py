from __future__ import annotations
from random import Random

from torch import Tensor
from torch.utils.data import Dataset
import torch
from ballfish import DistributionParams, create_distribution
from math import radians, pi, sin, cos


class PsfGeometricDataset(Dataset[Tensor]):
    def __init__(
        self,
        width: int,
        height: int,
        sphere_dpt: DistributionParams,
        cylinder_dpt: DistributionParams,
        angle_deg: DistributionParams,
        pupil_diameter_mm: DistributionParams,
        am2px: float = 0.001,
        seed: int = 42,
        size: int = 10000,
    ):
        self._width = width
        self._height = height
        self._seed = seed
        self._size = size
        self._sphere_dpt = create_distribution(sphere_dpt)
        self._cylinder_dpt = create_distribution(cylinder_dpt)
        self._angle_deg = create_distribution(angle_deg)
        self._pupil_diameter_mm = create_distribution(pupil_diameter_mm)
        self._am2px = am2px

        x = torch.arange(width, dtype=torch.float32)
        y = torch.arange(height, dtype=torch.float32)
        x = x - width * 0.5
        y = y - height * 0.5
        self._y, self._x = torch.meshgrid(y, x, indexing="ij")

    def geometric_generator(
        self,
        sphere_dpt: float,
        cylinder_dpt: float,
        angle_rad: float,
        pupil_diameter_mm: float,
    ) -> Tensor:
        """
        Create optic PSF for fixed viewing distance 400 cm and canvas size 50 cm.

        Loosely based on https://github.com/scottmsul/EyeSimulator/blob/master/Geometric_Optics_2_Defocus.ipynb
        """

        x, y, am2px = self._x, self._y, self._am2px

        blur_angle = pi * 0.5 + angle_rad

        # Radius of pupil
        r = pupil_diameter_mm * 0.5

        cos_theta = cos(blur_angle)
        sin_theta = sin(blur_angle)

        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y

        # Convert radius to angle minutes
        rad2am = 180.0 / pi * 60.0

        # Ellips semi-axis
        a = r * abs(sphere_dpt + cylinder_dpt) * rad2am * am2px
        b = r * abs(sphere_dpt) * rad2am * am2px

        # Distance map
        dist = torch.square(x_rot / a) + torch.square(y_rot / b)

        # Setting an ellips
        kernel = (dist <= 1).float()

        # Normalizing
        kernel /= kernel.sum()

        return kernel

    def __getitem__(self, index: int) -> Tensor:
        random = Random(f"{self._seed}|{index}")
        psf = self.geometric_generator(
            sphere_dpt=self._sphere_dpt(random),
            cylinder_dpt=self._cylinder_dpt(random),
            angle_rad=radians(self._angle_deg(random)),
            pupil_diameter_mm=self._pupil_diameter_mm(random),
        )
        return psf[None]

    def __len__(self):
        return self._size
