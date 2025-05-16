"""
Refactored version of https://github.com/J-FHu/UDCUNet

MIT License

Copyright (c) 2022 Hoven Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations
import functools
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from .download_path import download_path, PyOlimpHF


@torch.no_grad()
def default_init_weights(net_l: list[nn.Module], scale: float = 1.0) -> None:
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer_unet(block: Callable[[], nn.Module], n_layers: int):
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def make_layer(block: Callable[[], nn.Module], n_layers: int):
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        out = self.conv2(out)
        return identity + out


class SFTLayer(nn.Module):
    def __init__(
        self, in_nc: int = 32, out_nc: int = 64, nf: int = 32
    ) -> None:
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(
            F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.2, inplace=True)
        )
        shift = self.SFT_shift_conv1(
            F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.2, inplace=True)
        )
        return x[0] * (scale + 1) + shift


class ResBlock_with_SFT(nn.Module):
    def __init__(
        self, nf: int = 64, in_nc: int = 32, out_nc: int = 64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.out_nc = out_nc
        self.in_nc = in_nc
        self.sft1 = SFTLayer(in_nc=self.in_nc, out_nc=self.out_nc, nf=32)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=self.in_nc, out_nc=self.out_nc, nf=32)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])


class UDCUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 32,
        depths: list[int] = [2, 2, 2, 8, 2, 2, 2],
        DyK_size: int = 3,
    ) -> None:
        super().__init__()
        self.DyK_size = DyK_size

        ### Condition
        basic_Res = functools.partial(ResidualBlockNoBN, nf=nf)

        self.cond_head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        self.cond_first = make_layer_unet(basic_Res, 2)

        self.CondNet0 = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
        )

        self.CondNet1 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 1),
        )

        self.CondNet2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 4, 1),
        )

        self.CondNet3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 8, 1),
        )

        ## Kernel

        self.k_head = nn.Sequential(
            nn.Conv2d(in_nc + 5, nf, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        self.k_first = make_layer_unet(basic_Res, 2)

        self.KNet0 = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * self.DyK_size * self.DyK_size, 1),
        )

        self.KNet1 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2 * self.DyK_size * self.DyK_size, 1),
        )

        self.KNet2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 4 * self.DyK_size * self.DyK_size, 1),
        )

        self.KNet3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 8 * self.DyK_size * self.DyK_size, 1),
        )

        ## Base
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        basic_block = functools.partial(
            ResBlock_with_SFT, nf=nf, in_nc=nf, out_nc=nf
        )
        basic_block2 = functools.partial(
            ResBlock_with_SFT, nf=nf * 2, in_nc=nf * 2, out_nc=nf * 2
        )
        basic_block4 = functools.partial(
            ResBlock_with_SFT, nf=nf * 4, in_nc=nf * 4, out_nc=nf * 4
        )
        basic_block8 = functools.partial(
            ResBlock_with_SFT, nf=nf * 8, in_nc=nf * 8, out_nc=nf * 8
        )

        self.enconv_layer0 = make_layer(basic_block, depths[0])
        self.down_conv0 = nn.Conv2d(nf, nf * 2, 3, 2, 1)

        self.enconv_layer1 = make_layer(basic_block2, depths[1])
        self.down_conv1 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)

        self.enconv_layer2 = make_layer(basic_block4, depths[2])
        self.down_conv2 = nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)

        self.Bottom_conv = make_layer(basic_block8, depths[3])

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 4 * 4, 3, 1, 1), nn.PixelShuffle(2)
        )
        self.deconv_layer2 = make_layer(basic_block4, depths[4])

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 2 * 4, 3, 1, 1), nn.PixelShuffle(2)
        )
        self.deconv_layer1 = make_layer(basic_block2, depths[5])

        self.up_conv0 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.PixelShuffle(2)
        )
        self.deconv_layer0 = make_layer(basic_block, depths[6])

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    @classmethod
    def from_path(cls, path: PyOlimpHF, **kwargs: Any):
        model = cls(**kwargs).cuda()
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        # to run on original `net_g_600000.pth`, load state_dict["state"]
        model.load_state_dict(state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> tuple[Tensor, Tensor]:
        return image, psf

    def forward(self, x_psf: tuple[Tensor, Tensor]) -> tuple[Tensor]:
        x, psf = x_psf
        assert x.ndim == 4
        assert x.shape[1] == 3, x.shape[1]
        assert psf.ndim == 4
        assert psf.shape[1] == 1
        psf = psf.repeat((1, 5, 1, 1))  # view(1, 5, 1, 1)
        psf = psf.expand(x.shape[0], -1, x.shape[2], x.shape[3])
        k_fea = torch.cat((x, psf), 1)
        k_fea = self.k_first(self.k_head(k_fea))
        kernel0 = self.KNet0(k_fea)
        kernel1 = self.KNet1(k_fea)
        kernel2 = self.KNet2(k_fea)
        kernel3 = self.KNet3(k_fea)

        cond = self.cond_first(self.cond_head(x))
        cond0 = self.CondNet0(cond)
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.conv_first(x)

        fea0, _ = self.enconv_layer0((fea0, cond0))
        down0 = self.down_conv0(fea0)

        fea1, _ = self.enconv_layer1((down0, cond1))
        down1 = self.down_conv1(fea1)

        fea2, _ = self.enconv_layer2((down1, cond2))
        down2 = self.down_conv2(fea2)
        feaB, _ = self.Bottom_conv((down2, cond3))
        feaB = feaB + _kernel2d_conv(down2, kernel3, self.DyK_size)

        up2 = self.up_conv2(feaB) + _kernel2d_conv(
            fea2, kernel2, self.DyK_size
        )
        defea2, _ = self.deconv_layer2((up2, cond2))

        up1 = self.up_conv1(defea2) + _kernel2d_conv(
            fea1, kernel1, self.DyK_size
        )
        defea1, _ = self.deconv_layer1((up1, cond1))

        up0 = self.up_conv0(defea1) + _kernel2d_conv(
            fea0, kernel0, self.DyK_size
        )
        defea0, _ = self.deconv_layer0((up0, cond0))

        out = F.relu(x + self.conv_last(defea0))

        return (out,)

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    def arguments(self, *args):
        return {}


def _kernel2d_conv(feat_in: Tensor, kernel: Tensor, ksize: int) -> Tensor:
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(
        feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate"
    )
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(
        N, H, W, channels, ksize, ksize
    )
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, dim=-1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


def _demo():
    from ..._demo import demo

    def demo_unet_udc(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = UDCUNet()
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            progress(0.05)
            (precompensation,) = model(model.preprocess(image, psf))
            progress(1.0)
            return precompensation

    demo("UNET_UDC", demo_unet_udc, mono=False)


if __name__ == "__main__":
    _demo()
