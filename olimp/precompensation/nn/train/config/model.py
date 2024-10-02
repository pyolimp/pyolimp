from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field
from .base import StrictModel


class ModelConfig(StrictModel):
    pass


class VDSR(ModelConfig):
    name: Literal["vdsr"]

    def get_instance(self):
        from ...models.vdsr import VDSR

        return VDSR()


class VAE(ModelConfig):
    name: Literal["vae"]

    def get_instance(self):
        from ...models.vae import VAE

        return VAE()


class UNET_b0(ModelConfig):
    name: Literal["unet_b0"]

    def get_instance(self):
        from ...models.unet_efficient_b0 import PrecompensationUNETB0

        return PrecompensationUNETB0()


class PrecompensationUSRNet(ModelConfig):
    name: Literal["precompensationusrnet"]

    def get_instance(self):
        from ...models.usrnet import PrecompensationUSRNet

        return PrecompensationUSRNet()


class PrecompensationDWDN(ModelConfig):
    name: Literal["precompensationdwdn"]
    n_levels: int = 1

    def get_instance(self):
        from ...models.dwdn import PrecompensationDWDN

        return PrecompensationDWDN(n_levels=self.n_levels)


class Generator_transformer_pathch4_844_48_3_nouplayer_server5(ModelConfig):
    name: Literal["Generator_transformer_pathch4_844_48_3_nouplayer_server5"]

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_844_48_3_nouplayer_server5 import (
            Generator_transformer_pathch4_844_48_3_nouplayer_server5,
        )

        return Generator_transformer_pathch4_844_48_3_nouplayer_server5()


Model = Annotated[
    VDSR
    | VAE
    | UNET_b0
    | PrecompensationUSRNet
    | PrecompensationDWDN
    | Generator_transformer_pathch4_844_48_3_nouplayer_server5,
    Field(..., discriminator="name"),
]
