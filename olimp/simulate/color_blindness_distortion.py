from typing import Literal

import torch
from torch import Tensor

from olimp.simulate import Distortion


class ColorBlindnessDistortion(Distortion):
    def __init__(
        self,
        sim_type: Literal["Vienot", "Huang", "Farup"],
        blindness_type: Literal["PROTAN", "DEUTAN", "rg"],
    ) -> None:
        assert sim_type in [
            "Vienot",
            "Huang",
            "Farup",
        ], "no such distortion"
        assert blindness_type in [
            "PROTAN",
            "DEUTAN",
            "rg",
        ], "no such distortion"
        self.sim_type = sim_type
        self.blindness_type = blindness_type

    @staticmethod
    def _local_change_range(image: Tensor, quantile: float = 0.98):
        max_channel = torch.max(image, axis=2)[0]
        quantile_98 = torch.quantile(max_channel, quantile)
        divisor = quantile_98
        normalized_dichros = torch.clip(image / divisor, 0, 1)
        return normalized_dichros, divisor

    @staticmethod
    def _linearRGB_from_sRGB(image: Tensor) -> Tensor:
        # Convert sRGB to linearRGB (copied from daltonlens.convert.linearRGB_from_sRGB)
        out = torch.empty_like(image)
        small_mask = image < 0.04045
        large_mask = torch.logical_not(small_mask)
        out[small_mask] = image[small_mask] / 12.92
        out[large_mask] = torch.pow((image[large_mask] + 0.055) / 1.055, 2.4)
        return out

    @staticmethod
    def _sRGB_from_linearRGB(image: Tensor) -> Tensor:
        # Convert linearRGB to sRGB. Made on the basis of daltonlens.convert.sRGB_from_linearRGB
        # by Nicolas Burrus. Clipping operation was removed.
        out = torch.empty_like(image)
        small_mask = image < 0.0031308
        large_mask = torch.logical_not(small_mask)
        out[small_mask] = image[small_mask] * 12.92
        out[large_mask] = (
            torch.pow(image[large_mask], 1.0 / 2.4) * 1.055 - 0.055
        )
        return out

    @staticmethod
    def _convert_from_sRGB(sRGB: Tensor) -> Tensor:
        linRGB = ColorBlindnessDistortion._linearRGB_from_sRGB(sRGB)
        LMS_from_RGB = torch.tensor(
            [
                [0.27293945, 0.66418685, 0.06287371],
                [0.10022701, 0.78761123, 0.11216177],
                [0.01781695, 0.10961952, 0.87256353],
            ]
        )

        LMS = linRGB @ LMS_from_RGB.T
        return LMS

    @staticmethod
    def _convert_from_LMS(LMS: Tensor) -> Tensor:
        RGB_from_LMS = torch.tensor(
            [
                [5.30329968, -4.49954803, 0.19624834],
                [-0.67146001, 1.86248629, -0.19102629],
                [-0.0239335, -0.14210614, 1.16603964],
            ]
        )
        inv_linRGB = LMS @ RGB_from_LMS.T
        sRGB = ColorBlindnessDistortion._sRGB_from_linearRGB(inv_linRGB)
        return sRGB

    @staticmethod
    def _simulate(
        image: Tensor,
        sim_type: Literal["Vienot", "Huang", "Farup"],
        blindness_type: Literal["PROTAN", "DEUTAN", "rg"],
    ) -> Tensor:
        protan_Vienot = torch.tensor(
            [[0.0, 1.06481845, -0.06481845], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        protan_Huang = torch.tensor(
            [[0.0, 1.20166964, -0.20166964], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        deutan_Vienot = torch.tensor(
            [[1.0, 0.0, 0.0], [0.93912723, 0.0, 0.06087277], [0.0, 0.0, 1.0]]
        )
        deutan_Huang = torch.tensor(
            [[1.0, 0.0, 0.0], [0.83217547, 0.0, 0.16782453], [0.0, 0.0, 1.0]]
        )
        FarupRGmatrix = torch.tensor(
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
        )

        if blindness_type == "rg" and sim_type == "Farup":
            return image @ FarupRGmatrix.T
        elif blindness_type in ("PROTAN", "DEUTAN") and sim_type in (
            "Vienot",
            "Huang",
        ):
            lms = ColorBlindnessDistortion._convert_from_sRGB(image)
            if blindness_type == "PROTAN":
                if sim_type == "Vienot":
                    dichromat_LMS = lms @ protan_Vienot.T
                elif sim_type == "Huang":
                    dichromat_LMS = lms @ protan_Huang.T
            elif blindness_type == "DEUTAN":
                if sim_type == "Vienot":
                    dichromat_LMS = lms @ deutan_Vienot.T
                elif sim_type == "Huang":
                    dichromat_LMS = lms @ deutan_Huang.T
            sRGB = ColorBlindnessDistortion._convert_from_LMS(dichromat_LMS)
            sRGB = torch.clip(sRGB, 0, 1)
            return sRGB
        else:
            raise NotImplementedError

    def __call__(self, I: Tensor) -> Tensor:
        assert I.ndim > 2
        if I.ndim == 3:
            I = I[None]
        I_sim = torch.zeros_like(I, dtype=torch.float)
        for idx, image in enumerate(I):
            image = image - image.min()
            image = image / image.max()
            image_normalized, _ = ColorBlindnessDistortion._local_change_range(
                image.permute(1, 2, 0), 0.98
            )
            I_sim[idx] = ColorBlindnessDistortion._simulate(
                image_normalized, self.sim_type, self.blindness_type
            ).permute(2, 0, 1)
        return I_sim