from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor
import torch.nn.functional as F


class LCN(Module):
    def __init__(self, kernel_size: int = 7, invert: bool = False) -> None:
        super(LSM, self).__init__()  # type: ignore
        self.kernel_size = kernel_size
        self.invert = invert

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward method to compute similarity between two images after LCN.

        Parameters:
            x (Tensor): The first image in the format [B, C, H, W].
            y (Tensor): The second image in the format [B, C, H, W].

        Returns:
            Tensor: Similarity value between 0 and 1.
        """
        lcn_x = self.local_contrast_normalization(x)
        lcn_y = self.local_contrast_normalization(y)

        # Compute similarity score
        mse = torch.mean((lcn_x - lcn_y) ** 2)

        # Convert MSE to similarity (0 to 1 range, lower MSE means higher similarity)
        similarity = (
            torch.exp(-mse) if not self.invert else 1.0 - torch.exp(-mse)
        )

        return similarity

    def local_contrast_normalization(
        self, image: Tensor, epsilon: float = 1e-5
    ) -> Tensor:
        """
        Performs local contrast normalization on an image.

        Parameters:
            image (Tensor): Input image in the format [B, C, H, W].
            epsilon (float): A small value to avoid division by zero.

        Returns:
            Tensor: Normalized image.
        """
        # Ensure the input image has the format [B, C, H, W]
        assert (
            image.ndim == 4
        ), "Input image must be in the format [B, C, H, W]"

        # Create a kernel for mean and standard deviation computation
        kernel = torch.ones(
            (1, 1, self.kernel_size, self.kernel_size), device=image.device
        ) / (self.kernel_size**2)

        # Compute the local mean
        local_mean = F.conv2d(
            image, kernel, padding=self.kernel_size // 2, groups=image.size(1)
        )

        # Compute the squared difference from the mean for variance
        local_var = F.conv2d(
            (image - local_mean) ** 2,
            kernel,
            padding=self.kernel_size // 2,
            groups=image.size(1),
        )

        # Standard deviation (local contrast)
        local_std = torch.sqrt(local_var + epsilon)

        # Normalization: subtract mean and divide by standard deviation
        normalized_image = (image - local_mean) / local_std

        return normalized_image
