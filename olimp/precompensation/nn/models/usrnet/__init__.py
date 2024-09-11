from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor
import torchvision
from .model import USRNet


class PrecompensationUSRNet(USRNet):
    def __init__(
        self,
        n_iter: int = 8,
        h_nc: int = 64,
        in_nc: int = 4,
        out_nc: int = 3,
        nc: list[int] = [64, 128, 256, 512],
        nb: int = 2,
        act_mode: Literal[
            "C",
            "T",
            "B",
            "I",
            "R",
            "r",
            "L",
            "l",
            "2",
            "3",
            "4",
            "U",
            "u",
            "M",
            "A",
        ] = "R",  # activation function, see `.basicblock.conv`
        downsample_mode: Literal[
            "avgpool", "maxpool", "strideconv"
        ] = "strideconv",
        upsample_mode: Literal[
            "upconv", "pixelshuffle", "convtranspose"
        ] = "convtranspose",
    ):
        super().__init__(
            n_iter=n_iter,
            h_nc=h_nc,
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )
        # Add a Sigmoid layer to constrain the output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, k: Tensor, scale_factor: int, sigma: Tensor):
        x = super().forward(x, k, scale_factor, sigma)
        return self.sigmoid(x)

    @classmethod
    def from_path(cls, path: str, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> Tensor:
        psf = psf.unsqueeze(0).unsqueeze(0)
        resize_transform = torchvision.transforms.Resize((512, 512))

        image = resize_transform(image)
        img_gray = image.to(torch.float32)[None, None, ...]
        img_gray = torchvision.transforms.Resize((512, 512))(img_gray)
        # lower image contrast to make this demo look good
        img_gray = (img_gray * (0.7 - 0.3)) + 0.3

        return torch.cat(
            [img_gray, img_gray, img_gray], dim=1
        )  # B, 3, 512, 512

    def xxx():
        pass
