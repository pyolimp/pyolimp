from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field
from .base import StrictModel


class ModelConfig(StrictModel):
    pass


class VDSR(ModelConfig):
    name: Literal["vdsr"]
    path: str | None = Field(
        None, examples=["hf://RVI/vdsr.pth", "~/.weights/vdsr.pth"]
    )

    def get_instance(self):
        from ...models.vdsr import VDSR

        if self.path is not None:
            return VDSR.from_path(path=self.path)
        return VDSR()


class VAE(ModelConfig):
    name: Literal["vae"]
    path: str | None = Field(
        None, examples=["hf://RVI/vae.pth", "~/.weights/vae.pth"]
    )

    def get_instance(self):
        from ...models.vae import VAE

        if self.path is not None:
            return VAE.from_path(path=self.path)
        return VAE()


class CVAE(ModelConfig):
    name: Literal["cvae"]
    path: str | None = Field(
        None, examples=["hf://RVI/cvae.pth", "~/.weights/cvae.pth"]
    )

    def get_instance(self):
        from ...models.cvae import CVAE

        if self.path is not None:
            return CVAE.from_path(path=self.path)
        return CVAE()


class UNETVAE(ModelConfig):
    name: Literal["unetvae"]
    path: str | None = Field(
        None, examples=["hf://RVI/unetvae.pth", "~/.weights/unetvae.pth"]
    )

    def get_instance(self):
        from ...models.unetvae import UNETVAE

        if self.path is not None:
            return UNETVAE.from_path(path=self.path)
        return UNETVAE()


class UNET_b0(ModelConfig):
    name: Literal["unet_b0"]
    path: str | None = Field(
        None,
        examples=[
            "hf://RVI/unet-efficientnet-b0.pth",
            "~/.weights/unet-efficientnet-b0.pth",
        ],
    )
    in_channels: int = 3
    out_channels: int = 1

    def get_instance(self):
        from ...models.unet_efficient_b0 import PrecompensationUNETB0

        if self.path is not None:
            return PrecompensationUNETB0.from_path(
                path=self.path,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )
        return PrecompensationUNETB0(
            in_channels=self.in_channels, out_channels=self.out_channels
        )


class PrecompensationUSRNet(ModelConfig):
    name: Literal["precompensationusrnet"]
    path: str | None = Field(
        None, examples=["hf://RVI/usrnet.pth", "~/.weights/usrnet.pth"]
    )

    n_iter: int = 8
    h_nc: int = 64
    in_nc: int = 4
    out_nc: int = 3
    nc: list[int] = [64, 128, 256, 512]
    nb: int = 2

    def get_instance(self):
        from ...models.usrnet import PrecompensationUSRNet

        if self.path is not None:
            return PrecompensationUSRNet.from_path(path=self.path)
        return PrecompensationUSRNet(
            n_iter=self.n_iter,
            h_nc=self.h_nc,
            in_nc=self.in_nc,
            out_nc=self.out_nc,
            nc=self.nc,
            nb=self.nb,
        )


class PrecompensationDWDN(ModelConfig):
    name: Literal["precompensationdwdn"]
    n_levels: int = 1
    path: str | None = Field(
        None, examples=["hf://RVI/dwdn.pt", "~/.weights/dwdn.pt"]
    )

    def get_instance(self):
        from ...models.dwdn import PrecompensationDWDN

        if self.path is not None:
            return PrecompensationDWDN.from_path(path=self.path)
        return PrecompensationDWDN(n_levels=self.n_levels)


class CVDSwin3Channels(ModelConfig):
    name: Literal["cvd_swin_3channels"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
            "~/.weights/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.cvd_swin_3channels import (
            CVDSwin3Channels,
        )

        if self.path is not None:
            return CVDSwin3Channels.from_path(path=self.path)

        return CVDSwin3Channels()


class CVDSwin4Channels(ModelConfig):
    name: Literal["cvd_swin_4channels"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/cvd_swin_4channels.pth",
            "~/.weights/cvd_swin_4channels.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.cvd_swin_4channels import (
            CVDSwin4Channels,
        )

        if self.path is not None:
            return CVDSwin4Channels.from_path(path=self.path)

        return CVDSwin4Channels()


class CVDSwin1Channel(ModelConfig):
    name: Literal["cvd_swin_1channel"]
    path: str | None = Field(
        None,
        examples=[
            "hf://RVI/cvd_swin_1channel.pth",
            "~/.weights/cvd_swin_1channel.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.cvd_swin_1channel import (
            CVDSwin1Channel,
        )

        if self.path is not None:
            return CVDSwin1Channel.from_path(path=self.path)

        return CVDSwin1Channel()


class Generator_transformer_pathch4_8421_48_3_nouplayer_server5(ModelConfig):
    name: Literal["Generator_transformer_pathch4_8421_48_3_nouplayer_server5"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
            "~/.weights/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_8421_48_3_nouplayer_server5 import (
            Generator_transformer_pathch4_8421_48_3_nouplayer_server5,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_8421_48_3_nouplayer_server5.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_8421_48_3_nouplayer_server5()


class Generator_transformer_pathch4_844_48_3_server5(ModelConfig):
    name: Literal["Generator_transformer_pathch4_844_48_3_server5"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_844_48_3_server5.pth",
            "~/.weights/Generator_transformer_pathch4_844_48_3_server5.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_844_48_3_server5 import (
            Generator_transformer_pathch4_844_48_3_server5,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_844_48_3_server5.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_844_48_3_server5()


class Generator_transformer_pathch4_844_48_3(ModelConfig):
    name: Literal["Generator_transformer_pathch4_844_48_3"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_844_48_3.pth",
            "~/.weights/Generator_transformer_pathch4_844_48_3.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_844_48_3 import (
            Generator_transformer_pathch4_844_48_3,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_844_48_3.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_844_48_3()


class Generator_transformer_pathch4_844_48_3_nouplayer_server5(ModelConfig):
    name: Literal["Generator_transformer_pathch4_844_48_3_nouplayer_server5"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_844_48_3_nouplayer_server5.pth",
            "~/.weights/Generator_transformer_pathch4_844_48_3_nouplayer_server5.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_844_48_3_nouplayer_server5 import (
            Generator_transformer_pathch4_844_48_3_nouplayer_server5,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_844_48_3_nouplayer_server5.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_844_48_3_nouplayer_server5()


class Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton(
    ModelConfig
):
    name: Literal[
        "Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton"
    ]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton.pth",
            "~/.weights/Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton import (
            Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton.from_path(
                path=self.path
            )

        return (
            Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton()
        )


class Generator_transformer_pathch4_8_3_48_3(ModelConfig):
    name: Literal["Generator_transformer_pathch4_8_3_48_3"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_8_3_48_3.pth",
            "~/.weights/Generator_transformer_pathch4_8_3_48_3.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_8_3_48_3 import (
            Generator_transformer_pathch4_8_3_48_3,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_8_3_48_3.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_8_3_48_3()


class Generator_transformer_pathch4_6_3_48_3(ModelConfig):
    name: Literal["Generator_transformer_pathch4_6_3_48_3"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_6_3_48_3.pth",
            "~/.weights/Generator_transformer_pathch4_6_3_48_3.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_6_3_48_3 import (
            Generator_transformer_pathch4_6_3_48_3,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_6_3_48_3.from_path(
                path=self.path
            )

        return Generator_transformer_pathch4_6_3_48_3()


class Generator_transformer_pathch4_1_1(ModelConfig):
    name: Literal["Generator_transformer_pathch4_1_1"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch4_1_1.pth",
            "~/.weights/Generator_transformer_pathch4_1_1.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch4_1_1 import (
            Generator_transformer_pathch4_1_1,
        )

        if self.path is not None:
            return Generator_transformer_pathch4_1_1.from_path(path=self.path)

        return Generator_transformer_pathch4_1_1()


class Generator_transformer_pathch2(ModelConfig):
    name: Literal["Generator_transformer_pathch2"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch2.pth",
            "~/.weights/Generator_transformer_pathch2.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch2 import (
            Generator_transformer_pathch2,
        )

        if self.path is not None:
            return Generator_transformer_pathch2.from_path(path=self.path)

        return Generator_transformer_pathch2()


class Generator_cnn_pathch4_844_48_3_nouplayer_server5(ModelConfig):
    name: Literal["Generator_cnn_pathch4_844_48_3_nouplayer_server5"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
            "~/.weights/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_cnn_pathch4_844_48_3_nouplayer_server5 import (
            Generator_cnn_pathch4_844_48_3_nouplayer_server5,
        )

        if self.path is not None:
            return Generator_cnn_pathch4_844_48_3_nouplayer_server5.from_path(
                path=self.path
            )

        return Generator_cnn_pathch4_844_48_3_nouplayer_server5()


class Generator_transformer_pathch2_1_1(ModelConfig):
    name: Literal["Generator_transformer_pathch2_1_1"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch2_1_1.pth",
            "~/.weights/Generator_transformer_pathch2_1_1.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch2_1_1 import (
            Generator_transformer_pathch2_1_1,
        )

        if self.path is not None:
            return Generator_transformer_pathch2_1_1.from_path(path=self.path)

        return Generator_transformer_pathch2_1_1()


class Generator_transformer_pathch2_no_Unt(ModelConfig):
    name: Literal["Generator_transformer_pathch2_no_Unt"]
    path: str | None = Field(
        None,
        examples=[
            "hf://CVD/Generator_transformer_pathch2_no_Unt.pth",
            "~/.weights/Generator_transformer_pathch2_no_Unt.pth",
        ],
    )

    def get_instance(self):
        from ...models.cvd_swin.Generator_transformer_pathch2_no_Unt import (
            Generator_transformer_pathch2_no_Unt,
        )

        if self.path is not None:
            return Generator_transformer_pathch2_no_Unt.from_path(
                path=self.path
            )

        return Generator_transformer_pathch2_no_Unt()


Model = Annotated[
    VDSR
    | VAE
    | CVAE
    | UNETVAE
    | UNET_b0
    | PrecompensationUSRNet
    | PrecompensationDWDN
    | CVDSwin3Channels
    | CVDSwin4Channels
    | CVDSwin1Channel
    | Generator_transformer_pathch4_8421_48_3_nouplayer_server5
    | Generator_transformer_pathch4_844_48_3_server5
    | Generator_transformer_pathch4_844_48_3
    | Generator_transformer_pathch4_844_48_3_nouplayer_server5
    | Generator_transformer_pathch4_844_48_3_nouplayer_server5_no_normalizaiton
    | Generator_transformer_pathch4_8_3_48_3
    | Generator_transformer_pathch4_6_3_48_3
    | Generator_transformer_pathch4_1_1
    | Generator_transformer_pathch2
    | Generator_cnn_pathch4_844_48_3_nouplayer_server5
    | Generator_transformer_pathch2_1_1
    | Generator_transformer_pathch2_no_Unt,
    Field(..., discriminator="name"),
]
