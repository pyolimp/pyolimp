from __future__ import annotations
from typing import Literal, Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchvision
from olimp.processing import conv
from ..simulate.color_blindness_distortion import ColorBlindnessDistortion
from torchvision.transforms.v2 import Resize, Grayscale

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


def demo(
    name: Literal["Tennenholtz-Zachevsky"],
    opt_function: Callable[
        [Tensor, ColorBlindnessDistortion, Callable[[float], None]], Tensor
    ],
    distortion: ColorBlindnessDistortion,
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_l = progress.add_task("Load data", total=3)
        task_p = progress.add_task(name, total=1.0)

        progress.advance(task_l)
        img = torchvision.io.read_image("./tests/test_data/73.png")
        progress.advance(task_l)
        img = img / 255.0
        progress.advance(task_l)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.device(device):

            callback: Callable[[float], None] = lambda c: progress.update(
                task_p, completed=c
            )
            precompensation = opt_function(
                img.to(device), distortion, callback
            )
            cvd_procompensated = distortion(precompensation)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        dpi=72, figsize=(12, 9), ncols=2, nrows=2
    )
    assert img.shape[0] == 3
    img = img[0]
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    ax1.imshow(img)
    ax1.set_title("Original")
    ax2.imshow(img, vmin=0.0, vmax=1.0)
    ax2.set_title(f"Source ({img.min()}, {img.max()})")

    p_arr = precompensation.cpu().detach().numpy()
    ax3.set_title(
        f"Procompensation: {name} ({p_arr.min():g}, {p_arr.max():g})"
    )
    if p_arr.ndim == 3:
        p_arr = p_arr.transpose(1, 2, 0)
    ax3.imshow(p_arr, vmin=0.0, vmax=1.0)

    rp_arr = cvd_procompensated.cpu().detach().numpy()
    assert rp_arr.shape[0] == 1
    rp_arr = rp_arr[0]
    ax4.set_title(
        f"Retinal Procompensated ({rp_arr.min():g}, {rp_arr.max():g})"
    )
    if rp_arr.ndim == 3:
        rp_arr = rp_arr.transpose(1, 2, 0)
    ax4.imshow(rp_arr, vmin=0.0, vmax=1.0, cmap="gray")

    plt.show()
