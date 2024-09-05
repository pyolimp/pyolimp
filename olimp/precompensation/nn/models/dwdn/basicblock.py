import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1]
        )
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


"""
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
"""


# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel):
    ker_f = convert_psf2otf(
        kernel, img.size()
    )  # discrete fourier transform of kernel
    nsr = wiener_filter_para(img)
    denominator = inv_fft_kernel_est(ker_f, nsr)
    img1 = img
    numerator = torch.fft.fft2(img1)
    deblur = deconv(denominator, numerator)
    return deblur


# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = diff.shape[2] * diff.shape[2]
    mean_n = torch.sum(diff, (2, 3)).view(-1, 1, 1, 1) / num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2, 3)) / (num - 1)
    mean_input = torch.sum(_input_blur, (2, 3)).view(-1, 1, 1, 1) / num
    var_s2 = (
        torch.sum(
            (_input_blur - mean_input) * (_input_blur - mean_input), (2, 3)
        )
        / (num - 1)
    ) ** (0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1, 1, 1, 1)
    return NSR


# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    ker_f = ker_f.to(NSR.device)
    inv_denominator = (
        ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0]
        + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1]
        + NSR
    )
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return torch.view_as_complex(inv_ker_f)


# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = inv_ker_f * fft_input_blur
    deblur = torch.fft.ifft2(deblur_f)
    return deblur


# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size: tp.List[int]):
    psf = torch.zeros(size)
    # circularly shift
    centre = ker.shape[2] // 2
    psf[:, :, :centre, :centre] = ker[:, :, (centre):, (centre):]
    psf[:, :, :centre, -(centre):] = ker[:, :, (centre):, :(centre)]
    psf[:, :, -(centre):, :centre] = ker[:, :, :(centre), (centre):]
    psf[:, :, -(centre):, -(centre):] = ker[:, :, :(centre), :(centre)]
    otf = torch.view_as_real(torch.fft.fft2(psf))

    return otf
