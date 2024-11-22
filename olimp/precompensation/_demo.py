from __future__ import annotations
from typing import Literal, Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchvision
from olimp.processing import conv
from torchvision.transforms.v2 import Resize, Grayscale

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


def demo(
    name: Literal["Montalto", "Bregman Jumbo", "Huang", "Feng Xu"],
    opt_function: Callable[[Tensor, Tensor, Callable[[float], None]], Tensor],
    mono: bool = False,
    num_output_channels: int = 1,
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_l = progress.add_task("Load data", total=3)
        task_p = progress.add_task(name, total=1.0)

        psf_info = np.load("./tests/test_data/psf.npz")
        progress.advance(task_l)
        img = torchvision.io.read_image("./tests/test_data/horse.jpg")
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
            psf = torch.fft.fftshift(
                torch.tensor(psf_info["psf"]).to(torch.float32)
            )[None, None, ...]

            callback: Callable[[float], None] = lambda c: progress.update(
                task_p, completed=c
            )
            (precompensation,) = opt_function(img.to(device), psf, callback)
            retinal_procompensated = conv(precompensation, psf)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        dpi=72, figsize=(12, 9), ncols=2, nrows=2
    )
    ax1.imshow(psf_info["psf"])
    ax1.set_title(
        f"PSF (S={psf_info['S']}, C={psf_info['C']}, "
        f"A={psf_info['A']}, sum={psf_info['psf'].sum():g})"
    )
    assert img.shape[0] == 1
    img = img[0]
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    ax2.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Source ({img.min()}, {img.max()})")

    p_arr = precompensation.cpu().detach().numpy()
    ax3.set_title(
        f"Procompensation: {name} ({p_arr.min():g}, {p_arr.max():g})"
    )
    assert p_arr.shape[0] == 1
    p_arr = p_arr[0]
    if p_arr.ndim == 3:
        p_arr = p_arr.transpose(1, 2, 0)
    ax3.imshow(p_arr, vmin=0.0, vmax=1.0, cmap="gray")

    rp_arr = retinal_procompensated.cpu().detach().numpy()
    assert rp_arr.shape[0] == 1
    rp_arr = rp_arr[0]
    ax4.set_title(
        f"Retinal Procompensated ({rp_arr.min():g}, {rp_arr.max():g})"
    )
    if rp_arr.ndim == 3:
        rp_arr = rp_arr.transpose(1, 2, 0)
    ax4.imshow(rp_arr, vmin=0.0, vmax=1.0, cmap="gray")

    plt.show()
