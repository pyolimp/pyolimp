''' complex package for pytorch'''
import torch

def conj(x):
    dim = len(x.size()) - 1
    x_conj = torch.stack((x[..., 0], -x[..., 1]), dim=dim)
    return x_conj


def abs_square(x, keepdim=False):
    x_abs = (x[..., 0] ** 2 + x[..., 1] ** 2)
    if keepdim:
        dim = len(x.size()) - 1
        x_abs = torch.stack((x_abs, torch.zeros_like(x_abs)), dim=dim)
    return x_abs


def real(x):
    return x[..., 0]


def image(x):
    return x[..., 1]


def mul(x, y):
    dim = len(x.size()) - 1
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    mul = torch.stack((real, image), dim=dim)
    return mul


def div(x, y):
    dim = len(x.size()) - 1
    y_abs = y[..., 0] ** 2 + y[..., 1] ** 2
    real = (x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1]) / y_abs
    image = (x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]) / y_abs
    div = torch.stack((real, image), dim=dim)
    return div


def fft(x):
    dim = len(x.size())
    if dim == 2:
        # Fx = torch.rfft(x, signal_ndim=2, onesided=False)
        Fx = torch.view_as_real(torch.fft.fft2(x, dim=(-2, -1), norm='backward'))
        return Fx
    elif dim == 3:
        chan_num = x.size()[0]
        Fx = torch.zeros(*x.size(), 2).cuda()
        for i in range(chan_num):
            # Fx[i,] = torch.rfft(x[i,], signal_ndim=2, onesided=False)
            Fx[i] = torch.view_as_real(torch.fft.fft2(x[i], dim=(-2, -1), norm='backward'))
        return Fx


def ifft(Fx):
    dim = len(Fx.size())
    if dim == 3:
        # x = torch.irfft(Fx, signal_ndim=2, onesided=False)
        x = torch.fft.irfft2(torch.view_as_complex(Fx), s=(Fx.size(-3), Fx.size(-2)), dim=(-2, -1), norm='backward')
        return x
    elif dim == 4:
        im_num = Fx.size()[0]
        x = torch.zeros(*Fx.size()[:-1])
        for i in range(im_num):
            # x[i,] = torch.irfft(Fx[i,], signal_ndim=2, onesided=False)
            x[i] = torch.fft.irfft2(torch.view_as_complex(Fx[i,]), s=(Fx.size(-2), Fx.size(-1)), dim=(-2, -1),norm='backward')
        return x
  