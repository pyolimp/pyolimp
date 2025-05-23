from __future__ import annotations
from typing import NamedTuple
import torch
from torch import nn, Tensor
from .utils import comfft as cf
from .dual_path_unet import dual_path_unet
from .utils.wavelet import generate_wavelet, wv_norm, Fwv


class KerUncArgs(NamedTuple):
    lmds: list[float]
    layers: int
    deep: int


class kernel_error_model(nn.Module):
    def __init__(self, args: KerUncArgs):
        super().__init__()
        self.args = args
        self.dec2d, _ = generate_wavelet(1, dim=2)
        norm = torch.from_numpy(wv_norm(self.dec2d)).cuda()
        lmd: list[Tensor] = []
        for i in range(len(args.lmds)):
            lmd.append(torch.ones(len(self.dec2d)) * args.lmds[i] / norm)

        self.net = nn.ModuleList()
        self.net = self.net.append(Db_Inv(lmd=lmd[0]))
        for i in range(args.layers):
            self.net = self.net.append(DP_Unet())
            self.net = self.net.append(
                Dn_CNN(depth=args.deep, in_chan=(i + 1))
            )
            self.net = self.net.append(Db_Inv(lmd=lmd[i + 1]))

    def forward(self, y: Tensor, Fker: Tensor) -> Tensor:
        # intermediate results stored in xhat and z by lists.
        xhat = [None] * (self.args.layers + 1)
        z = [None] * (self.args.layers)
        u = [None] * (self.args.layers)

        xhat[0] = self.net[0](y, Fker, None, None)
        u[0] = self.net[1](xhat[0], y, Fker)
        z[0] = self.net[2](xhat[0])

        for i in range(self.args.layers - 1):
            xhat[i + 1] = self.net[3 * i + 3](y, Fker, z[i], u[i])
            u[i + 1] = self.net[3 * i + 4](xhat[i], y, Fker)
            input = torch.cat([xhat[j] for j in range(0, i + 2)], dim=1)
            z[i + 1] = self.net[3 * i + 5](input)

        i = self.args.layers - 1
        xhat[i + 1] = self.net[3 * i + 3](y, Fker, z[i], u[i])
        return xhat


class Db_Inv(nn.Module):
    def __init__(self, lmd: Tensor):
        super().__init__()
        self.dec2d, _ = generate_wavelet(frame=1)
        self.chn_num = len(self.dec2d)
        self.lmd = lmd.view(self.chn_num, 1, 1, 1).cuda()

    def forward(
        self,
        y: Tensor,
        Fker: Tensor,
        z: Tensor | None = None,
        u: Tensor | None = None,
    ) -> Tensor:
        if z is None:
            z = torch.zeros_like(y)
        if u is None:
            u = torch.zeros_like(y)

        im_num = y.shape[0]
        xhat = torch.zeros_like(y)

        for i in range(im_num):
            shape = y[
                i,
                0,
            ].size()[-2:]
            Fw = Fwv(self.dec2d, shape=shape).cuda()

            Fker_conj = cf.conj(Fker[i]).cuda()
            Fw_conj = cf.conj(Fw).cuda()

            Fy = cf.fft(
                y[
                    i,
                    0,
                ]
                - u[
                    i,
                    0,
                ]
            )  # minus w to incorporate the prior approximation of noise
            Fz = cf.fft(
                z[
                    i,
                    0,
                ]
            ).cuda()

            Fx_num = cf.mul(Fker_conj, Fy) + torch.sum(
                self.lmd * cf.mul(Fw_conj, cf.mul(Fw, Fz)), dim=0
            )
            Fx_den = cf.abs_square(Fker[i], keepdim=True) + torch.sum(
                self.lmd * cf.mul(Fw_conj, Fw), dim=0
            )
            Fx = cf.div(Fx_num, Fx_den)
            xhat[
                i,
                0,
            ] = cf.ifft(Fx)
        return xhat


class Dn_CNN(nn.Module):
    """Plain CNN"""

    def __init__(
        self,
        depth: int = 17,
        n_channels: int = 64,
        in_chan: int = 1,
        out_chan: int = 1,
        in_conv: int = 1,
        out_conv: int = 1,
    ) -> None:
        super().__init__()
        kernel_size = 3
        padding = 1
        layers: list[nn.Module] = []
        miss_conv = int(in_conv == 0) + int(out_conv == 0)

        if in_conv == 0:
            layers.append(
                nn.Conv2d(
                    in_channels=in_chan,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(
                nn.BatchNorm2d(
                    n_channels,
                    eps=0.0001,
                    momentum=0.95,
                    track_running_stats=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        else:
            for _ in range(in_conv):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_chan,
                        out_channels=n_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=True,
                    )
                )
                layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - in_conv - out_conv - miss_conv):
            layers.append(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(
                nn.BatchNorm2d(
                    n_channels,
                    eps=0.0001,
                    momentum=0.95,
                    track_running_stats=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))

        if out_conv == 0:
            layers.append(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=out_chan,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(
                nn.BatchNorm2d(
                    out_chan,
                    eps=0.0001,
                    momentum=0.95,
                    track_running_stats=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        else:
            for _ in range(out_conv):
                layers.append(
                    nn.Conv2d(
                        in_channels=n_channels,
                        out_channels=out_chan,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    )
                )

        self.cnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self.cnn(x)
        return out

    def _initialize_weights(self) -> None:
        import torch.nn.init as init

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class DP_Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = dual_path_unet()
        self._initialize_weights()

    def forward(self, x: Tensor, y: Tensor, Fker: Tensor) -> Tensor:
        im_num = x.size(0)
        kx = torch.zeros_like(x)
        for i in range(im_num):
            Fz = cf.fft(
                x[
                    i,
                    0,
                ]
            )
            kx[
                i,
                0,
            ] = cf.ifft(cf.mul(Fker[i], Fz))
        res = y - kx
        u = self.net(res, x)
        return u

    def _initialize_weights(self) -> None:
        import torch.nn.init as init

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
