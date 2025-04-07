from __future__ import annotations
import torch
from torch import nn
from torch import Tensor

# import torchvision
from olimp.processing import fft_conv
from .download_path import download_path, PyOlimpHF


class _ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, (3, 3), (1, 1), (1, 1), bias=False
        )
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    """
    .. image:: ../../../../_static/vdsr.svg
       :class: full-width
    """

    def __init__(self) -> None:
        super().__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(_ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    @classmethod
    def from_path(cls, path: PyOlimpHF):
        model = cls()
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def forward(self, x: Tensor) -> tuple[Tensor]:
        identity = x[:, :1, :, :]

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return (self.sigmoid(out),)

    def preprocess(self, image: Tensor, psf: Tensor) -> Tensor:
        # image = torchvision.transforms.Grayscale()(image)
        # img_gray = torchvision.transforms.Resize((512, 512))(image)
        # img_gray = image.to(torch.float32)
        img_blur = fft_conv(image, psf)

        return torch.cat(
            [
                image,
                img_blur,
                psf,
            ],
            dim=1,
        )

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    def arguments(self, *args):
        return {}


def _demo():
    from ..._demo import demo
    from typing import Callable

    def demo_vdsr(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = VDSR.from_path("hf://RVI/vdsr.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            (precompensation,) = model(inputs)
            progress(1.0)
            return precompensation

    demo("VDSR", demo_vdsr, mono=True)


if __name__ == "__main__":
    _demo()
