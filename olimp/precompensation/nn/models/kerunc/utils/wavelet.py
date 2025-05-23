"""wavelet toolbox for multilevel wavelet decomposition and reconstruction"""

from __future__ import annotations

import numpy as np
import torch


def GenerateFilter(frame=1):
    if frame == 0:
        # D = [np.array([0,1/2,1/2]),np.array([0,1/2,-1/2])]
        D = np.array([[0, 1 / 2, 1 / 2], [0, 1 / 2, -1 / 2]])
    elif frame == 1:
        D = np.array(
            [
                [1 / 4, 1 / 2, 1 / 4],
                [1 / 4 * np.sqrt(2), 0, -1 / 4 * np.sqrt(2)],
                [-1 / 4, 1 / 2, -1 / 4],
            ]
        )
    elif frame == 3:
        D = np.array(
            [
                [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16],
                [1 / 8, 2 / 8, 0 / 8, -2 / 8, -1 / 8],
                [
                    -1 / 16 * np.sqrt(6),
                    0,
                    2 / 16 * np.sqrt(6),
                    0,
                    -1 / 16 * np.sqrt(6),
                ],
                [-1 / 8, 2 / 8, 0, -2 / 8, 1 / 8],
                [1 / 16, -4 / 16, 6 / 16, -4 / 16, 1 / 16],
            ]
        )
    return D


def DecFilterML(D, level=1, dim=2):
    nD = len(D)
    lD = len(D[1])
    Dec = {}
    for i in range(nD):
        Dec[1, i] = np.flip(D[i], 0)

    for lv in range(2, level + 1):
        step = 2 ** (lv - 1)
        for i in range(1, nD + 1):
            H0 = np.zeros(shape=[(lD - 1) * step + 1])
            H0[0 : step * (lD - 1) + 1 : step] = Dec[1, i - 1]
            Dec[lv, i - 1] = np.convolve(Dec[lv - 1, 0], H0, mode="full")

    if dim == 1:
        return Dec
    elif dim == 2:
        Dec2 = {}
        for lv in range(1, level + 1):
            for i in range(0, nD):
                for j in range(0, nD):
                    Dec2[lv, i, j] = np.kron(
                        np.reshape(Dec[lv, i], (-1, 1)),
                        np.reshape(Dec[lv, j], (1, -1)),
                    )
        return Dec2
    elif dim == 3:
        Dec3 = {}
        for lv in range(1, level + 1):
            for i in range(0, nD):
                for j in range(0, nD):
                    for k in range(0, nD):
                        tmp = np.kron(
                            np.reshape(Dec[lv, i], (-1, 1, 1)),
                            np.reshape(Dec[lv, j], (1, -1, 1)),
                        )
                        Dec3[lv, i, j, k] = np.kron(
                            tmp, np.reshape(Dec[lv, k], (1, 1, -1))
                        )
        return Dec3


def RecFilterML(D, level=1, dim=2):
    nD = len(D)
    lD = len(D[1])
    Rec = {}
    for lv in range(1, level + 1):
        if lv == 1:
            for i in range(0, nD):
                Rec[1, i] = D[i]
        else:
            step = 2 ** (lv - 1)
            for i in range(0, nD):
                H0 = np.zeros(shape=[(lD - 1) * step + 1])
                H0[0 : step * (lD - 1) + 1 : step] = Rec[1, i]
                Rec[lv, i] = np.convolve(Rec[lv - 1, 0], H0, mode="full")

    if dim == 1:
        return Rec
    elif dim == 2:
        Rec2 = {}
        for lv in range(1, level + 1):
            for i in range(0, nD):
                for j in range(0, nD):
                    Rec2[lv, i, j] = np.kron(
                        np.reshape(Rec[lv, i], (1, -1)),
                        np.reshape(Rec[lv, j], (-1, 1)),
                    ).T
        return Rec2
    elif dim == 3:
        Rec3 = {}
        for lv in range(1, level + 1):
            for i in range(0, nD):
                for j in range(0, nD):
                    for k in range(0, nD):
                        tmp = np.kron(
                            np.reshape(Rec[lv, i], (-1, 1, 1)),
                            np.reshape(Rec[lv, j], (1, -1, 1)),
                        )
                        Rec3[lv, i, j, k] = np.kron(
                            tmp, np.reshape(Rec[lv, k], (1, 1, -1))
                        )
        return Rec3


def generate_wavelet(frame=1, dim=2, highpass=True):
    D = GenerateFilter(frame)
    if dim == 2:
        Dec = DecFilterML(D)
        Rec = RecFilterML(D)
        if highpass:
            del Dec[1, 0, 0]
            del Rec[1, 0, 0]
    elif dim == 3:
        Dec = DecFilterML(D, dim=3)
        Rec = RecFilterML(D, dim=3)
        if highpass:
            del Dec[1, 0, 0, 0]
            del Rec[1, 0, 0, 0]
    return Dec, Rec


def wv_norm(Dec, dtype=np.float32):
    chan = len(Dec)
    WvNorm = np.zeros(chan, dtype=dtype)
    j = 0
    for d in Dec:
        norm = np.sum(np.abs(Dec[d]))
        WvNorm[j] = norm
        j += 1
    return WvNorm


def for_fft(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int32)
    ker_mat[: ker_shape[0], : ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(0, 1))
    return ker_mat


def Fwv(Dec, shape=(256, 256)):
    chan_num = len(Dec)
    W = np.zeros((chan_num, *shape), dtype=np.float32)
    i = 0
    for d in Dec:
        W[i,] = for_fft(Dec[d], shape=shape)
        i += 1

    W = torch.from_numpy(W)
    Fw = torch.zeros((chan_num, *shape, 2))
    for i in range(chan_num):
        Fw[i,] = torch.view_as_real(
            torch.fft.fft2(W[i,], dim=(-2, -1), norm="backward")
        )
    return Fw
