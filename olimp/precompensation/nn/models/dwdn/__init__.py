from __future__ import annotations
import torch
from torch import nn, Tensor
import torchvision

from .model import DWDN


class PrecompensationDWDN(DWDN):
    def __init__(self, n_levels: int = 1, scale: float = 0):
        super().__init__(n_levels=n_levels, scale=scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image: Tensor, psf: Tensor) -> Tensor:
        image = super().forward(image, psf)[0]
        return self.sigmoid(image)

    @classmethod
    def from_path(cls, path: str, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> Tensor:
        resize_transform = torchvision.transforms.Resize((512, 512))

        image = resize_transform(image)
        img_gray = image.to(torch.float32)[None, None, ...]
        img_gray = torchvision.transforms.Resize((512, 512))(img_gray)
        # lower image contrast to make this demo look good
        img_gray = (img_gray * (0.7 - 0.3)) + 0.3

        return torch.cat(
            [img_gray, img_gray, img_gray], dim=1
        )  # B, 3, 512, 512

    def arguments(self, input: Tensor, psf: Tensor) -> dict[str, None]:
        return {"psf": torch.fft.fftshift(psf.unsqueeze(0).unsqueeze(0))}
