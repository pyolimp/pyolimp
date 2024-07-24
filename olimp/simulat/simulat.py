import torch
from abc import ABC, abstractmethod


class Simulator(ABC):
    @abstractmethod
    def fast_conv(self, I: torch.Tensor, K: torch.Tensor, return_2d: bool = False) -> torch.Tensor:
        """
        Perform convolution using FFT on the input tensor I with the kernel tensor K.

        Parameters:
        I (torch.Tensor): Input image tensor of shape (b, n, w, h), (c, w, h), or (w, h) with torch.float data type.
        K (torch.Tensor): PSF kernel tensor of shape (b, 1, w, h), (1, w, h), or (w, h) with torch.float data type.
        return_2d (bool): Whether to return a 2D tensor if input is 2D. Default is False.

        Returns:
        torch.Tensor: Convolved tensor with the same spatial dimensions as input tensor I.
        """
        pass


class RetinalImageSimulator(Simulator):
    def fast_conv(self, I: torch.Tensor, K: torch.Tensor, return_2d: bool = False) -> torch.Tensor:
        """
        Perform convolution using FFT on the input tensor I with the kernel tensor K.

        Parameters:
        I (torch.Tensor): Input image tensor of shape (b, n, w, h), (c, w, h), or (w, h) with torch.float data type.
        K (torch.Tensor): PSF kernel tensor of shape (b, 1, w, h), (1, w, h), or (w, h) with torch.float data type.
        return_2d (bool): Whether to return a 2D tensor if input is 2D. Default is False.

        Returns:
        torch.Tensor: Convolved tensor with the same spatial dimensions as input tensor I.
        """
        # Check data type
        assert I.dtype == torch.float, "I must have data type torch.float"
        assert K.dtype == torch.float, "K must have data type torch.float"

        # Check value range
        assert torch.all((I >= 0) & (I <= 1)), "Values in I must be in the range [0, 1]"
        assert torch.all((K >= 0) & (K <= 1)), "Values in K must be in the range [0, 1]"

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
        assert I.shape[2] == K.shape[2] and I.shape[3] == K.shape[3], "The width and height of I and K must match"

        # Handling batch size
        if I.shape[0] != K.shape[0]:
            if K.shape[0] == 1:
                K = K.expand(I.shape[0], -1, -1, -1)
            else:
                assert False, "The batch size of I and K must match or K must have batch size 1"

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
        if original_ndim == 2 and return_2d:
            conv_result = conv_result.squeeze(0).squeeze(0)

        return conv_result



if __name__ == "__main__":
    simulator = RetinalImageSimulator()

    I_2d = torch.rand(64, 64, dtype=torch.float)
    K_2d = torch.rand(64, 64, dtype=torch.float)
    result_2d = simulator.fast_conv(I_2d, K_2d, return_2d=True)
    print(result_2d.shape)  # Expected shape (64, 64)

    I_2d = torch.rand(64, 64, dtype=torch.float)
    K_2d = torch.rand(64, 64, dtype=torch.float)
    result_2d = simulator.fast_conv(I_2d, K_2d)
    print(result_2d.shape)  # Expected shape (1, 1, 64, 64)

    I_3d = torch.rand(3, 64, 64, dtype=torch.float)
    K_3d = torch.rand(1, 64, 64, dtype=torch.float)
    result_3d = simulator.fast_conv(I_3d, K_3d)
    print(result_3d.shape)  # Expected shape (1, 3, 64, 64)

    I_4d = torch.rand(2, 3, 64, 64, dtype=torch.float)
    K_4d = torch.rand(2, 1, 64, 64, dtype=torch.float)
    result_4d = simulator.fast_conv(I_4d, K_4d)
    print(result_4d.shape)  # Expected shape (2, 3, 64, 64)