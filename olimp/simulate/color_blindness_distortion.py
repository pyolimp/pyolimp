from typing import Literal

import torch
from torch import Tensor

from olimp.evaluation.cs.linrgb import linRGB
from olimp.evaluation.cs.srgb import sRGB

from olimp.simulate import ApplyDistortion


class ColorBlindnessDistortion:
    """
    .. image:: ../_static/color_blindness_distortion.svg
       :class: full-width
    """

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
            sim_matrix = torch.tensor(
                (
                    (0.0, 1.06481845, -0.06481845),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
        elif blindness_type == "deutan":
            sim_matrix = torch.tensor(
                (
                    (1.0, 0.0, 0.0),
                    (0.93912723, 0.0, 0.06087277),
                    (0.0, 0.0, 1.0),
                )
            )
        else:
            raise KeyError("no such distortion")

        self.sim_matrix = (
            self.RGB_from_LMS.to(sim_matrix.device)
            @ sim_matrix
            @ self.LMS_from_RGB.to(sim_matrix.device)
        )

        self.blindness_type = blindness_type

    @staticmethod
    def _linearRGB_from_sRGB(image: Tensor) -> Tensor:
        return linRGB().from_sRGB(image)

    @staticmethod
    def _sRGB_from_linearRGB(image: Tensor) -> Tensor:
        return sRGB().from_linRGB(image)

    @classmethod
    def _simulate(cls, image: Tensor, sim_matrix: Tensor) -> Tensor:
        linRGB = cls._linearRGB_from_sRGB(image)
        dichromat_LMS = torch.tensordot(
            sim_matrix.to(image.device), linRGB, dims=1
        )
        return cls._sRGB_from_linearRGB(dichromat_LMS).clip_(0.0, 1.0)

    def __call__(self) -> ApplyDistortion:
        return self.apply

    def apply(self, image: Tensor):
        assert image.ndim == 4, image.ndim
        image_sim = torch.zeros_like(image, dtype=torch.float)
        for image, out in zip(image, image_sim):
            out[:] = self._simulate(image, self.sim_matrix)
        return image_sim


def _demo():
    from ._demo_distortion import demo

    def demo_simulate():
        yield ColorBlindnessDistortion("protan")(), "protan"
        yield ColorBlindnessDistortion("deutan")(), "deutan"

    demo("ColorBlindnessDistortion", demo_simulate)


if __name__ == "__main__":
    _demo()
