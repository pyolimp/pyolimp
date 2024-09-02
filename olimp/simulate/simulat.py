from typing import Literal, TYPE_CHECKING, NamedTuple

import torch
from abc import ABC, abstractmethod

# if TYPE_CHECKING:
from torch import Tensor


class CVDSimulationParams(NamedTuple):
    sim_name: Literal["Vienot", "Huang", "Farup"]
    cvd_type: Literal["PROTAN", "DEUTAN", "rg"]


def simulate(image: Tensor, params: Tensor | CVDSimulationParams) -> Tensor:
    if isinstance(params, Tensor):
        sim = SimulateRetinalImage()
    elif isinstance(params, CVDSimulationParams):
        sim = SimulateDichromacy()
    else:
        raise ValueError(params)
    return sim(image, params)


class Simulate(ABC):
    @abstractmethod
    def __call__(
        self, image: Tensor, params: Tensor | CVDSimulationParams
    ) -> Tensor:
        raise NotImplementedError

    # @abstractmethod
    # def fast_conv(
    #     self, I: Tensor, K: Tensor, return_2d: bool = False
    # ) -> Tensor:
    #     """
    #     Perform convolution using FFT on the input tensor I with the kernel tensor K.

    #     Parameters:
    #     I (Tensor): Input image tensor of shape (b, n, w, h), (c, w, h), or (w, h) with torch.float data type.
    #     K (Tensor): PSF kernel tensor of shape (b, 1, w, h), (1, w, h), or (w, h) with torch.float data type.
    #     return_2d (bool): Whether to return a 2D tensor if input is 2D. Default is False.

    #     Returns:
    #     Tensor: Convolved tensor with the same spatial dimensions as input tensor I.
    #     """
    #     pass


class SimulateRetinalImage(Simulate):
    def __call__(self, I: Tensor, K: Tensor) -> Tensor:
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
        assert K.dtype == torch.float, "K must have data type torch.float"

        # Check value range
        assert torch.all(
            (I >= 0) & (I <= 1)
        ), "Values in I must be in the range [0, 1]"
        assert torch.all(
            (K >= 0) & (K <= 1)
        ), "Values in K must be in the range [0, 1]"

        original_ndim = I.ndim  # Store the original number of dimensions

        # Handle different input shapes
        if I.ndim == 2 and K.ndim == 2:
            I = I.unsqueeze(0).unsqueeze(0)  # Convert (w, h) to (1, 1, w, h)
            K = K.unsqueeze(0).unsqueeze(0)  # Convert (w, h) to (1, 1, w, h)
        elif I.ndim == 3 and K.ndim == 3:
            I = I.unsqueeze(0)  # Convert (c, w, h) to (1, c, w, h)
            K = K.unsqueeze(0)  # Convert (1, w, h) to (1, 1, w, h)

        # Check dimensions after potential unsqueezing
        assert I.ndim == 4, "I must be a 4-dimensional tensor (b, n, w, h)"
        assert K.ndim == 4, "K must be a 4-dimensional tensor (b, 1, w, h)"
        assert (
            I.shape[2] == K.shape[2] and I.shape[3] == K.shape[3]
        ), "The width and height of I and K must match"

        # Handling batch size
        if I.shape[0] != K.shape[0]:
            if K.shape[0] == 1:
                K = K.expand(I.shape[0], -1, -1, -1)
            else:
                assert (
                    False
                ), "The batch size of I and K must match or K must have batch size 1"

        b, n, w, h = I.shape

        # Expand K to match the channel dimension of I
        K_expanded = K.expand(-1, n, -1, -1)

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


class SimulateDichromacy(Simulate):
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
        linRGB = SimulateDichromacy._linearRGB_from_sRGB(sRGB)
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
        sRGB = SimulateDichromacy._sRGB_from_linearRGB(inv_linRGB)
        return sRGB

    @staticmethod
    def _simulate(image: Tensor, cvd_params: CVDSimulationParams) -> Tensor:
        assert cvd_params.sim_name in [
            "Vienot",
            "Huang",
            "Farup",
        ], "no such simulation"
        assert cvd_params.cvd_type in [
            "PROTAN",
            "DEUTAN",
            "rg",
        ], "no such simulation"
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

        if cvd_params.cvd_type == "rg" and cvd_params.sim_name == "Farup":
            return image @ FarupRGmatrix.T
        else:
            lms = SimulateDichromacy._convert_from_sRGB(image)
            if cvd_params.cvd_type == "PROTAN":
                if cvd_params.sim_name == "Vienot":
                    dichromat_LMS = lms @ protan_Vienot.T
                elif cvd_params.sim_name == "Huang":
                    dichromat_LMS = lms @ protan_Huang.T
            elif cvd_params.cvd_type == "DEUTAN":
                if cvd_params.sim_name == "Vienot":
                    dichromat_LMS = lms @ deutan_Vienot.T
                elif cvd_params.sim_name == "Huang":
                    dichromat_LMS = lms @ deutan_Huang.T
            sRGB = SimulateDichromacy._convert_from_LMS(dichromat_LMS)
            sRGB = torch.clip(sRGB, 0, 1)
            return sRGB

    def __call__(
        self, batch: Tensor, cvd_params: CVDSimulationParams
    ) -> Tensor:
        if batch.ndim == 3:
            batch = batch[None]
        batch_sim = torch.zeros_like(batch, dtype=torch.float)
        for idx, image in enumerate(batch):
            image = image - image.min()
            image = image / image.max()
            image_normalized, _ = SimulateDichromacy._local_change_range(
                image, 0.98
            )
            batch_sim[idx] = SimulateDichromacy._simulate(
                image_normalized, cvd_params
            )
            # breakpoint()
        return batch_sim


if __name__ == "__main__":
    I_2d = torch.rand(64, 64, dtype=torch.float)
    K_2d = torch.rand(64, 64, dtype=torch.float)
    result_2d = simulate(I_2d, K_2d)
    print(result_2d.shape)  # Expected shape (64, 64)

    I_2d = torch.rand(64, 64, dtype=torch.float)
    K_2d = torch.rand(64, 64, dtype=torch.float)
    result_2d = simulate(I_2d, K_2d)
    print(result_2d.shape)  # Expected shape (1, 1, 64, 64)

    I_3d = torch.rand(3, 64, 64, dtype=torch.float)
    K_3d = torch.rand(1, 64, 64, dtype=torch.float)
    result_3d = simulate(I_3d, K_3d)
    print(result_3d.shape)  # Expected shape (1, 3, 64, 64)

    I_4d = torch.rand(2, 3, 64, 64, dtype=torch.float)
    K_4d = torch.rand(2, 1, 64, 64, dtype=torch.float)
    result_4d = simulate(I_4d, K_4d)
    print(result_4d.shape)  # Expected shape (2, 3, 64, 64)
