from __future__ import annotations
from typing import Literal, TypeAlias, Any
import torch
from torch import nn, Tensor
from .model import USRNet
from ..download_path import download_path, PyOlimpHF

Input: TypeAlias = tuple[Tensor, Tensor, int, Tensor]


class PrecompensationUSRNet(USRNet):
    """
    Deep unfolding super-resolution network

    .. image:: ../../../../_static/usrnet.svg
       :class: full-width
    """

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

    def forward(self, inputs: Input):
        x, k, scale_factor, sigma = inputs
        x = super().forward(x, k, scale_factor, sigma)
        return (self.sigmoid(x),)

    def preprocess(
        self,
        image: Tensor,
        psf: Tensor,
        scale_factor: int = 1,
        noise_level: int = 0,
    ) -> Input:
        sigma = torch.tensor(noise_level).float().view([1, 1, 1, 1])
        sigma = sigma.repeat([image.shape[0], 1, 1, 1])
        psf = torch.fft.fftshift(psf)
        return image, psf, scale_factor, sigma

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    @classmethod
    def from_path(cls, path: PyOlimpHF, **kwargs: Any):
        model = cls(**kwargs)
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        return model

    def arguments(self, *args):
        return {}
