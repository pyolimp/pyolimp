import typing as tp

import torch
from torch import nn

from .utils import comfft as cf
from .dual_path_unet import dual_path_unet
from .utils.wavelet import generate_wavelet, wv_norm, Fwv


class KernelErrorModel(nn.Module):
    def __init__(
        self,
        lmds: tp.List[float],
        layers: int,
        deep: int,
    ):
        """Init KernelErrorModel.

        Parameters
        ----------
        lmds : int
            _description_
        layers : int
            Net layers count.
        deep : int
           One module deep.
        """
        super().__init__()
        self.layers = layers

        self.dec2d, _ = generate_wavelet(1, dim=2)
        norm = torch.from_numpy(wv_norm(self.dec2d))
        lmd = []
        for i in range(len(lmds)):
            lmd.append(torch.ones(len(self.dec2d)) * lmds[i] / norm)

        self.net = nn.ModuleList()
        self.net = self.net.append(Db_Inv(lmd = lmd[0]))
        for i in range(layers):
            self.net = self.net.append(DP_Unet())
            self.net = self.net.append(Dn_CNN(depth=deep, in_chan=(i+1)))
            self.net = self.net.append(Db_Inv(lmd=lmd[i+1]))

    def forward(self, y, Fker):
        # intermediate results stored in xhat and z by lists.
        xhat = [None] * (self.layers+1)
        z = [None] * (self.layers)
        u =  [None] * (self.layers)

        xhat[0] = self.net[0](y, Fker, None, None)
        u[0]  = self.net[1](xhat[0], y, Fker)
        z[0] = self.net[2](xhat[0])

        for i in range(self.layers-1):
            xhat[i+1] = self.net[3*i+3](y,  Fker, z[i], u[i])
            u[i+1] = self.net[3*i+4](xhat[i], y, Fker)
            input = torch.cat([xhat[j] for j in range(0,i+2)], dim = 1)
            z[i+1] = self.net[3*i+5](input)

        i = self.layers - 1
        xhat[i+1] = self.net[3*i+3](y, Fker, z[i], u[i])
        return xhat


class Db_Inv(nn.Module):
    def __init__(self, lmd):
        super(Db_Inv,self).__init__()
        self.dec2d, _ = generate_wavelet(frame=1)
        self.chn_num = len(self.dec2d)
        self.lmd = lmd.view(self.chn_num, 1, 1, 1).cpu()

    def forward(self, y, Fker, z=None, u=None):
        if z is None: z = torch.zeros_like(y, device=y.device)
        if u is None: u = torch.zeros_like(y, device=y.device)

        im_num = y.shape[0]
        xhat = torch.zeros_like(y, device=y.device)

        for i in range(im_num):
            shape = y[i,0,].size()[-2:]
            Fw = Fwv(self.dec2d, shape=shape).cpu()

            Fker_conj = cf.conj(Fker[i]).cpu()
            Fw_conj = cf.conj(Fw).cpu()

            Fy = cf.fft(y[i,0,] - u[i,0,])  # minus w to incorporate the prior approximation of noise
            Fz = cf.fft(z[i,0,]).cpu()

            Fx_num = cf.mul(Fker_conj, Fy).to(y.device) + torch.sum(self.lmd * cf.mul(Fw_conj, cf.mul(Fw, Fz)), dim=0).to(y.device)
            Fx_den = cf.abs_square(Fker[i], keepdim=True).to(y.device) + torch.sum(self.lmd * cf.mul(Fw_conj, Fw), dim=0).to(y.device)
            Fx = cf.div(Fx_num, Fx_den)
            xhat[i,0,] = cf.ifft(Fx)
        return xhat

class Dn_CNN(nn.Module):
    ''' Plain CNN '''
    def __init__(self, depth=17, n_channels=64, in_chan=1, out_chan=1, in_conv = 1, out_conv = 1):
        super(Dn_CNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        miss_conv = int(in_conv == 0) + int(out_conv == 0)

        if in_conv == 0:
            layers.append(nn.Conv2d(in_channels=in_chan, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
        else:
            for _ in range(in_conv):
                layers.append(nn.Conv2d(in_channels=in_chan, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                              bias=True))
                layers.append(nn.ReLU(inplace=True))


        for _ in range(depth-in_conv-out_conv-miss_conv):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))



        if out_conv == 0:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_chan, eps=0.0001, momentum = 0.95, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
        else:
            for _ in range(out_conv):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, kernel_size=kernel_size, padding=padding,bias=False))


        self.cnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.cnn(x)
        return out

    def _initialize_weights(self):
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
    def __init__(self):
        super(DP_Unet, self).__init__()
        self.net = dual_path_unet()
        self._initialize_weights()

    def forward(self, x, y, Fker):
        im_num = x.size(0)
        kx = torch.zeros_like(x)
        for i in range(im_num):
            Fz = cf.fft(x[i,0,])
            kx[i,0,] = cf.ifft(cf.mul(Fker[i], Fz))
        res = y - kx
        u = self.net(res, x)
        return u

    def _initialize_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)