from __future__ import annotations

import torch
from torch import Tensor
from matplotlib import pylab as plt
from math import pi

from olimp.simulate.psf_sca import PSFSCA


def show_one(image: Tensor, title: str) -> None:
    if image.isnan().any():
        raise ValueError("has nan")
    fig, ax1 = plt.subplots(dpi=72, figsize=(6, 4.5), ncols=1, nrows=1)
    plt.title(title)
    ax1.imshow(image)
    plt.show()


if __name__ == "__main__":

    params = {
        "sphere_dpt": -1.0,
        "cylinder_dpt": 0.0,
        "angle_rad": 0.0,
        "pupil_diameter_mm": 4.0,
        "am2px": 0.001,
    }

    params1 = {**params}
    show_one(PSFSCA(64, 64)(**params1), f"{params1}")
    params2 = {**params, "cylinder_dpt": 3}
    show_one(PSFSCA(64, 64)(**params2), f"{params2}")
    params3 = {**params, "cylinder_dpt": -2, "angle_rad": pi / 1.5}
    show_one(PSFSCA(64, 64)(**params3), f"{params3}")
