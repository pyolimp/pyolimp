from typing import Literal

import torch
from torch import Tensor

from olimp.evaluation.cs.linrgb import linRGB
from olimp.evaluation.cs.srgb import sRGB

from olimp.simulate import ApplyDistortion


class ColorBlindnessDistortion:
    LMS_from_RGB = torch.tensor(
        (
            (0.27293945, 0.66418685, 0.06287371),
            (0.10022701, 0.78761123, 0.11216177),
            (0.01781695, 0.10961952, 0.87256353),
        )
    )

    RGB_from_LMS = torch.tensor(
        (
            (5.30329968, -4.49954803, 0.19624834),
            (-0.67146001, 1.86248629, -0.19102629),
            (-0.0239335, -0.14210614, 1.16603964),
        ),
    )

    def __init__(
        self,
        blindness_type: Literal["protan", "deutan"],
    ) -> None:

        if blindness_type == "protan":
            self.sim_matrix = torch.tensor(
                (
                    (0.0, 1.06481845, -0.06481845),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
        elif blindness_type == "deutan":
            self.sim_matrix = torch.tensor(
                (
                    (1.0, 0.0, 0.0),
                    (0.93912723, 0.0, 0.06087277),
                    (0.0, 0.0, 1.0),
                )
            )
        else:
            raise KeyError("no such distortion")

        self.blindness_type = blindness_type

    @staticmethod
    def _linearRGB_from_sRGB(image: Tensor) -> Tensor:
        return linRGB().from_sRGB(image)

    @staticmethod
    def _sRGB_from_linearRGB(image: Tensor) -> Tensor:
        return sRGB().from_linRGB(image)

    @classmethod
    def _sRGB_to_LMS(cls, sRGB: Tensor) -> Tensor:
        linRGB = cls._linearRGB_from_sRGB(sRGB)
        return torch.tensordot(
            cls.LMS_from_RGB.to(sRGB.device),
            linRGB,
            dims=1,
        )

    @classmethod
    def _LMS_to_sRGB(cls, LMS: Tensor) -> Tensor:
        inv_linRGB = torch.tensordot(
            cls.RGB_from_LMS.to(LMS.device), LMS, dims=1
        )
        return cls._sRGB_from_linearRGB(inv_linRGB)

    @classmethod
    def _simulate(cls, image: Tensor, sim_matrix: Tensor) -> Tensor:
        lms = cls._sRGB_to_LMS(image)
        dichromat_LMS = torch.tensordot(
            sim_matrix.to(image.device), lms, dims=1
        )
        return cls._LMS_to_sRGB(dichromat_LMS).clip_(0.0, 1.0)

    def __call__(self) -> ApplyDistortion:
        return self.apply

    def apply(self, image: Tensor):
        assert image.ndim == 4, image.ndim
        image_sim = torch.zeros_like(image, dtype=torch.float)
        for image, out in zip(image, image_sim):
            out[:] = self._simulate(image, self.sim_matrix)
        return image_sim
