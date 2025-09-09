from __future__ import annotations

import torch
from torch import Tensor
from matplotlib import pylab as plt
from math import pi

from olimp.simulate.psf_gauss import PSFGauss


def show_one(image: Tensor, title: str) -> None:
    if image.isnan().any():
        raise ValueError("has nan")
    fig, ax1 = plt.subplots(dpi=72, figsize=(6, 4.5), ncols=1, nrows=1)
    plt.title(title)
    ax1.imshow(image)
    plt.show()


if __name__ == "__main__":

    params = {
        "center_x": 32,
        "center_y": 32,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 5.0,
    }

    show_one(PSFGauss(64, 64)(**params), f"{params}")
    params2 = {**params, "sigma_x": 6.0, "sigma_y": 3.0}
    show_one(PSFGauss(64, 64)(**params2), f"{params2}")
    params3 = {**params, "sigma_x": 6.0, "sigma_y": 3.0, "theta": pi / 5}
    show_one(PSFGauss(64, 64)(**params3), f"{params3}")
