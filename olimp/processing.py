from __future__ import annotations
import torch


def conv(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    assert (
        image.shape[-2:] == kernel.shape[-2:]
    ), f"Expected equal shapes, got: image={image.shape[-2:]}, kernel={kernel.shape[-2:]}"

    return torch.real(
        torch.fft.ifft2(torch.fft.fft2(image) * torch.fft.fft2(kernel))
    )


def scale_value(
    arr: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    black, white = arr.min(), arr.max()
    if black == white:
        mul = 0.0
        if min_val <= black <= max_val:
            add = black
        else:
            add = (
                min_val
                if abs(black - min_val) <= abs(black - max_val)
                else max_val
            )
    else:
        mul = (max_val - min_val) / (white - black)
        add = -black * mul + min_val
    out = torch.mul(arr, mul)
    out += add
    return out
