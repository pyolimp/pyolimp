from __future__ import annotations
from typing import Literal, Callable
import sys
from math import degrees
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from olimp.processing import fft_conv, resize_kernel
from olimp.demo_data import horse, psf as demo_psf
from torchvision.transforms.v2 import Resize, Grayscale

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


def demo(
    name: Literal[
        "Bregman Jumbo",
        "CVAE",
        "DWDN",
        "Feng Xu",
        "Global Tone Mapping",
        "Half-Quadratic",
        "Huang",
        "Ji",
        "KerUnc",
        "Montalto (FISTA)",
        "Montalto",
        "UNET_UDC",
        "UNET",
        "UNETVAE",
        "USRNET",
        "VAE",
        "VDSR",
    ],
    opt_function: Callable[[Tensor, Tensor, Callable[[float], None]], Tensor],
    mono: bool = False,
    num_output_channels: int = 1,
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        disable="--no-progress" in sys.argv,
    ) as progress:
        task_l = progress.add_task("Load data", total=3)
        task_p = progress.add_task(name, total=1.0)
        psf_sca = demo_psf()
        progress.advance(task_l)
        img = horse()
        progress.advance(task_l)
        img = img / 255.0
        img = Resize((512, 512))(img)
        if mono:
            img = Grayscale(num_output_channels=num_output_channels)(img)[
                None, ...
            ]
        else:
            img = img[None, ...]
        progress.advance(task_l)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.device(device):
            psf = psf_sca["psf"].to(torch.float32).to(device)
            psf = resize_kernel(psf[None, None, ...], img.shape[-2:])
            psf /= psf.sum()
            psf_shifted = torch.fft.fftshift(psf)

            callback: Callable[[float], None] = lambda c: progress.update(
                task_p, completed=c
            )
            (precompensation_0,) = opt_function(
                img.to(device), psf_shifted, callback
            )
            retinal_precompensated = fft_conv(precompensation_0, psf_shifted)

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        dpi=72, figsize=(12, 9), ncols=2, nrows=2
    )
    ax1.imshow(psf_sca["psf"])
    ax1.set_title(
        f"PSF (S={psf_sca['S']}, C={psf_sca['C']}, "
        f"A={degrees(psf_sca['A']):g}°, sum={psf_sca['psf'].sum():g})"
    )
    assert img.shape[0] == 1
    img = img[0]
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    ax2.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Source ({img.min()}, {img.max()})")

    p_arr = precompensation_0.detach().cpu().numpy()
    ax3.set_title(
        f"Procompensation: {name} ({p_arr.min():g}, {p_arr.max():g})"
    )
    if p_arr.ndim == 3:
        p_arr = p_arr.transpose(1, 2, 0)
    ax3.imshow(p_arr, vmin=0.0, vmax=1.0, cmap="gray")

    rp_arr = retinal_precompensated.cpu().detach().numpy()
    assert rp_arr.shape[0] == 1, rp_arr.shape[0]
    rp_arr = rp_arr[0]
    ax4.set_title(
        f"Retinal Precompensated ({rp_arr.min():g}, {rp_arr.max():g})"
    )
    if rp_arr.ndim == 3:
        rp_arr = rp_arr.transpose(1, 2, 0)
    ax4.imshow(rp_arr, vmin=0.0, vmax=1.0, cmap="gray")

    plt.show()
