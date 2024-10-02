import torch
from torch import Tensor

from olimp.simulate import Distortion


class RefractionDistortion(Distortion):
    def __init__(self, K: Tensor) -> None:
        assert K.dtype == torch.float, "K must have data type torch.float"
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
        elif original_ndim == 3:
            conv_result = conv_result.squeeze(0)
        return conv_result
