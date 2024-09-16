from typing import Literal, TYPE_CHECKING, Any

import torch
from abc import ABC, abstractmethod

# if TYPE_CHECKING:
from torch import Tensor


class Distortion(ABC):
    @abstractmethod
    def __init__(self, *args: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, I: Tensor) -> Tensor:
        raise NotImplementedError


class RefractionDistortion(Distortion):
    def __init__(self, K: Tensor) -> None:
        assert self.K.dtype == torch.float, "K must have data type torch.float"
        self.K = K

    def __call__(self, I: Tensor) -> Tensor:
        """
        Perform convolution using FFT on the input tensor I with the kernel tensor K.

        Parameters:
        I (Tensor): Input image tensor of shape (b, n, w, h), (c, w, h), or (w, h) with torch.float data type.
        K (Tensor): PSF kernel tensor of shape (b, 1, w, h), (1, w, h), or (w, h) with torch.float data type.
        return_2d (bool): Whether to return a 2D tensor if input is 2D. Default is False.

        Returns:
        Tensor: Convolved tensor with the same spatial dimensions as input tensor I.
        """
        # Check data type
        assert I.dtype == torch.float, "I must have data type torch.float"

        # Check value range
        assert torch.all(
            (I >= 0) & (I <= 1)
        ), "Values in I must be in the range [0, 1]"
        assert torch.all(
            (self.K >= 0) & (self.K <= 1)
        ), "Values in K must be in the range [0, 1]"

        original_ndim = I.ndim  # Store the original number of dimensions

        # Handle different input shapes
        if I.ndim == 2 and self.K.ndim == 2:
            I = I.unsqueeze(0).unsqueeze(0)  # Convert (w, h) to (1, 1, w, h)
            self.K = self.K.unsqueeze(0).unsqueeze(
                0
            )  # Convert (w, h) to (1, 1, w, h)
        elif I.ndim == 3 and self.K.ndim == 3:
            I = I.unsqueeze(0)  # Convert (c, w, h) to (1, c, w, h)
            self.K = self.K.unsqueeze(0)  # Convert (1, w, h) to (1, 1, w, h)

        # Check dimensions after potential unsqueezing
        assert I.ndim == 4, "I must be a 4-dimensional tensor (b, n, w, h)"
        assert (
            self.K.ndim == 4
        ), "K must be a 4-dimensional tensor (b, 1, w, h)"
        assert (
            I.shape[2] == self.K.shape[2] and I.shape[3] == self.K.shape[3]
        ), "The width and height of I and K must match"

        # Handling batch size
        if I.shape[0] != self.K.shape[0]:
            if self.K.shape[0] == 1:
                self.K = self.K.expand(I.shape[0], -1, -1, -1)
            else:
                assert (
                    False
                ), "The batch size of I and K must match or K must have batch size 1"

        b, n, w, h = I.shape

        # Expand K to match the channel dimension of I
        K_expanded = self.K.expand(-1, n, -1, -1)

        # Apply FFT to both the image tensor and the PSF tensor
        I_fft = torch.fft.fft2(I)
        K_fft = torch.fft.fft2(K_expanded)

        # Element-wise multiplication in the frequency domain
        conv_fft = I_fft * K_fft

        # Apply inverse FFT to obtain the convolution result in the spatial domain
        conv_result = torch.fft.ifft2(conv_fft)

        # Return the real part of the convolution result
        conv_result = torch.real(conv_result)

        # If the original input was 2D and return_2d is True, return a 2D tensor
        if original_ndim == 2:
            conv_result = conv_result.squeeze(0).squeeze(0)

        return conv_result


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
