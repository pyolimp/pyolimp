''' Image Tools '''
import numpy as np
from scipy.ndimage import filters
from scipy.signal import fftconvolve
import torch
import torch.nn.functional as F


def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif not x_tensor.is_cuda:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.detach().cpu().numpy()
        return x
    
    
def for_fft(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(0, 1))
    return ker_mat


def fspecial(type, *args):
    dtype = np.float32
    if type == 'average':
        siz = (args[0],args[0])
        h = np.ones(siz) / np.prod(siz)
        return h.astype(dtype)
    elif type == 'gaussian':
        p2 = args[0]
        p3 = args[1]
        siz = np.array([(p2[0]-1)/2 , (p2[1]-1)/2])
        std = p3
        x1 = np.arange(-siz[1], siz[1] + 1, 1)
        y1 = np.arange(-siz[0], siz[0] + 1, 1)
        x, y = np.meshgrid(x1, y1)
        arg = -(x*x + y*y) / (2*std*std)
        h = np.exp(arg)
        sumh = sum(map(sum, h))
        if sumh != 0:
            h = h/sumh
        return h.astype(dtype)
    elif type == 'motion':
        p2 = args[0]
        p3 = args[1]
        len = max(1, p2)
        half = (len - 1) / 2
        phi = np.mod(p3, 180) / 180 * np.pi

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        linewdt = 1

        eps = np.finfo(float).eps
        sx = np.fix(half * cosphi + linewdt * xsign - len * eps)
        sy = np.fix(half * sinphi + linewdt - len * eps)

        x1 = np.arange(0, sx + 1, xsign)
        y1 = np.arange(0, sy + 1, 1)
        x, y = np.meshgrid(x1, y1)

        dist2line = (y * cosphi - x * sinphi)
        rad = np.sqrt(x * x + y * y)

        lastpix = np.logical_and(rad >= half, np.abs(dist2line) <= linewdt)
        lastpix.astype(int)
        x2lastpix = half * lastpix - np.abs((x * lastpix + dist2line * lastpix * sinphi) / cosphi)
        dist2line = dist2line * (-1 * lastpix + 1) + np.sqrt(dist2line ** 2 + x2lastpix ** 2) * lastpix
        dist2line = linewdt + eps - np.abs(dist2line)
        logic = dist2line < 0
        dist2line = dist2line * (-1 * logic + 1)

        h1 = np.rot90(dist2line, 2)
        h1s = np.shape(h1)
        h = np.zeros(shape=(h1s[0] * 2 - 1, h1s[1] * 2 - 1))
        h[0:h1s[0], 0:h1s[1]] = h1
        h[h1s[0] - 1:, h1s[1] - 1:] = dist2line
        h = h / sum(map(sum, h)) + eps * len * len

        if cosphi > 0:
            h = np.flipud(h)

        return h.astype(dtype)


def cconv_np(data, ker):
    return filters.convolve(data, ker, mode='wrap')


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def cconv_torch(x,ker):
    with torch.no_grad():
        x_h, x_v = x.size()
        conv_ker = np.flip(np.flip(ker, 0), 1)
        ker = torch.FloatTensor(conv_ker.copy()).cuda()
        k_h, k_v = ker.size()
        k2_h = k_h // 2
        k2_v = k_v // 2
        x = torch.cat((x[-k2_h:,:], x, x[0:k2_h,:]), dim = 0).cuda()
        x = torch.cat((x[:,-k2_v:], x, x[:,0:k2_v]), dim = 1).cuda()
        x = x.unsqueeze(0).cuda()
        x = x.unsqueeze(1).cuda()
        ker = ker.unsqueeze(0).cuda()
        ker = ker.unsqueeze(1).cuda()
        y1 = F.conv2d(x, ker).cuda()
        y1 = torch.squeeze(y1)
        y = y1[-x_h:, -x_v:]
    return y


def pad_for_kernel(img, kernel, mode):
    p = [(d - 1) // 2 for d in kernel.shape]
    padding = [p, p]
    img_pad = np.pad(img, padding, mode)
    return img_pad


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, axis = 1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha * img + (1 - alpha) * blurred
    return img
