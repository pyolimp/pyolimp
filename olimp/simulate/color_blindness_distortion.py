from typing import Literal

import torch
from torch import Tensor

from olimp.evaluation.cs.linrgb import linRGB
from olimp.evaluation.cs.srgb import sRGB

from olimp.simulate import ApplyDistortion, Distortion


class ColorBlindnessDistortion(Distortion):
    """
    .. image:: ../_static/color_blindness_distortion.svg
       :class: full-width
    """

    # Anchors for simulation planes
    anchor_B = torch.Tensor(
        [0, 0, 1]
    )  # Blue anchor color for white-blue-yellow plane
    anchor_tritan = torch.Tensor(
        [0, 1, 0.5]
    )  # Anchor color for perpendicular plane

    RGB_from_LMS = torch.Tensor(
        (
            (5.30329968, -4.49954803, 0.19624834),
            (-0.67146001, 1.86248629, -0.19102629),
            (-0.0239335, -0.14210614, 1.16603964),
        ),
    )

    def __init__(self, hue_angle_deg: float | Tensor | None) -> None:
        if hue_angle_deg is not None:  # is None, when angle is dynamic
            assert isinstance(hue_angle_deg, (float, int)), hue_angle_deg
            with torch.device("cpu"):
                self.sim_matrix = self._get_simulation_tensor(hue_angle_deg)

    @classmethod
    def from_type(cls, blindness_type: Literal["protan", "deutan", "tritan"]):
        hue_angle = {"protan": 0, "deutan": 120, "tritan": 240}.get(
            blindness_type, None
        )
        if hue_angle is None:
            raise KeyError(f"no such distortion {blindness_type}")
        return cls(hue_angle)

    @staticmethod
    def _linearRGB_from_sRGB(image: Tensor) -> Tensor:
        return linRGB().from_sRGB(image)

    @staticmethod
    def _sRGB_from_linearRGB(image: Tensor) -> Tensor:
        return sRGB().from_linRGB(image)

    @classmethod
    def _get_simulation_matrix(cls, hue_angle_deg: float):
        """
        Code is based on Paul Maximov's <pmaximov@iitp.ru> work:
        https://github.com/PaulMaximov/general-dichromat-simulation
        """
        anchor_W_ort = torch.full((3,), 1 / 3**0.5)
        N_1plane_RGB_ort = cls._plane_normal_from_vectors(
            anchor_W_ort, cls.anchor_B
        )
        N_2plane_RGB_ort = cls._plane_normal_from_vectors(
            anchor_W_ort, cls.anchor_tritan
        )
        dichvec_LMS_ort = cls._confusion_vector_from_hue(hue_angle_deg)
        dichvec_RGB_ort = cls.RGB_from_LMS @ dichvec_LMS_ort
        dichvec_RGB_ort = dichvec_RGB_ort / torch.linalg.norm(dichvec_RGB_ort)

        sinfi_RGB_1plane = torch.abs(dichvec_RGB_ort @ N_1plane_RGB_ort)
        sinfi_RGB_2plane = torch.abs(dichvec_RGB_ort @ N_2plane_RGB_ort)

        if sinfi_RGB_1plane > sinfi_RGB_2plane:
            simmatr = cls._simmatr_from_points(
                anchor_W_ort, cls.anchor_B, dichvec_RGB_ort
            )
        else:
            simmatr = cls._simmatr_from_points(
                anchor_W_ort, cls.anchor_tritan, dichvec_RGB_ort
            )
        return simmatr

    @classmethod
    def _get_simulation_tensor(
        cls, hue_angle_deg: Tensor | float | int
    ) -> Tensor:
        if isinstance(hue_angle_deg, (float, int)):
            return cls._get_simulation_matrix(hue_angle_deg)[None]
        with torch.device("cpu"):
            sim_matrix: list[Tensor] = []
            for hue_angle in hue_angle_deg:
                sim_matrix.append(cls._get_simulation_matrix(hue_angle.item()))
            return torch.stack(sim_matrix)

    @staticmethod
    def _plane_normal_from_vectors(
        anchor1: torch.Tensor, anchor2: torch.Tensor
    ) -> torch.Tensor:
        """Function for calculation of normal unit vector of dichromatic projection
        plane from origin and two anchor points (Vienot et al., 1999)"""
        plane_normal = torch.empty(3)
        plane_normal[0] = anchor1[1] * anchor2[2] - anchor1[2] * anchor2[1]
        plane_normal[1] = anchor1[2] * anchor2[0] - anchor1[0] * anchor2[2]
        plane_normal[2] = anchor1[0] * anchor2[1] - anchor1[1] * anchor2[0]
        return plane_normal / torch.linalg.norm(plane_normal)

    @staticmethod
    def _simmatr_from_points(
        anchor1: torch.Tensor,
        anchor2: torch.Tensor,
        confusionvec: torch.Tensor,
    ) -> torch.Tensor:
        matrix = torch.vstack((confusionvec, anchor1, anchor2))
        matrix_inv = torch.linalg.inv(matrix)
        initial_values = torch.vstack((torch.zeros(3), anchor1, anchor2))
        return (matrix_inv @ initial_values).T

    @staticmethod
    def _confusion_vector_from_hue(hue_angle_deg: float) -> torch.Tensor:
        angle = torch.deg2rad(torch.tensor(hue_angle_deg))

        vec = torch.tensor((torch.cos(angle), torch.sin(angle), 1))
        matrix = torch.tensor(((2, 0, 1), (-1, 3**0.5, 1), (-1, -(3**0.5), 1)))
        return 1 / 3 * matrix @ vec

    @classmethod
    def _simulate(cls, image: Tensor, sim_matrix: Tensor) -> Tensor:
        linRGB = cls._linearRGB_from_sRGB(image)
        dichromat_LMS = torch.einsum(
            "bij,bjhw->bihw", sim_matrix.to(image.device), linRGB
        )
        return cls._sRGB_from_linearRGB(dichromat_LMS).clip(0.0, 1.0)

    def __call__(
        self, hue_angle_deg: float | Tensor | None = None
    ) -> ApplyDistortion:
        if hue_angle_deg is None:
            assert self.sim_matrix is not None
            sim_matrix = self.sim_matrix
        else:
            sim_matrix = self._get_simulation_tensor(hue_angle_deg)
        return lambda image: self._simulate(image, sim_matrix)


def _demo():
    from ._demo_distortion import demo

    def demo_simulate():
        yield ColorBlindnessDistortion.from_type("protan")(), "protan"
        yield ColorBlindnessDistortion.from_type("deutan")(), "deutan"
        yield ColorBlindnessDistortion.from_type("tritan")(), "tritan"

    demo("ColorBlindnessDistortion", demo_simulate)


if __name__ == "__main__":
    _demo()
