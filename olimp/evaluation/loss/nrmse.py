from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor


class NormalizedRootMSE(Module):
    """
    Normalized Root Mean Squared Error (NRMSE) loss in PyTorch.

    Args:
        normalization (str): The normalization type. Options are 'euclidean', 'min-max', or 'mean'.
        invert (bool): If True, returns `1 - NRMSE` for compatibility with certain metrics.
    """

    def __init__(
        self, normalization: str = "euclidean", invert: bool = True
    ) -> None:
        super(NormalizedRootMSE, self).__init__()  # type: ignore
        self.normalization = normalization.lower()
        self.invert = invert

        if self.normalization not in ["euclidean", "min-max", "mean"]:
            raise ValueError(
                "Unsupported normalization type. Choose from 'euclidean', 'min-max', or 'mean'."
            )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Normalized Root Mean Squared Error (NRMSE) between two tensors.

        Args:
            x (Tensor): Ground truth tensor.
            y (Tensor): Predicted tensor.

        Returns:
            Tensor: The computed NRMSE value. If invert is True, returns `1 - NRMSE`.
        """
        # Compute MSE
        mse_value = torch.mean((x - y) ** 2)

        # Compute normalization denominator
        if self.normalization == "euclidean":
            denom = torch.sqrt(torch.mean(x**2))
        elif self.normalization == "min-max":
            denom = x.max() - x.min()
        elif self.normalization == "mean":
            denom = x.mean()
        else:
            raise ValueError("Unsupported normalization type.")

        # Avoid division by zero
        if denom == 0:
            raise ValueError("Denominator for normalization is zero.")

        # Compute NRMSE
        nrmse_value = torch.sqrt(mse_value) / denom
        return 1 - nrmse_value if self.invert else nrmse_value
