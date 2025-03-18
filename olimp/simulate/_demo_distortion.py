from __future__ import annotations
from typing import Literal, Callable, Iterable
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms.v2 import Resize
from pathlib import Path

from olimp.simulate import ApplyDistortion


def demo(
    name: Literal["ColorBlindnessDistortion", "RefractionDistortion"],
    sim_functions: Callable[[], Iterable[tuple[ApplyDistortion, str]]],
    on: Literal["73.png", "horse.jpg"] = "73.png",
    size: tuple[int, int] = (256, 256),
):
    root = Path(__file__).parents[2]

    img = torchvision.io.read_image(str(root / f"tests/test_data/{on}"))[None]
    img = img / 255.0
    img = Resize(size)(img)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulation: list[torch.Tensor] = []
    labels: list[str] = []
    with torch.device(device):
        for func, label in sim_functions():
            out = func(img.to(device))
            simulation.append(out)
            labels.append(label)

    ncols = len(simulation) + 1
    _fig, axis = plt.subplots(
        dpi=72, figsize=(4 * ncols, 4), ncols=ncols, nrows=1
    )
    assert img.shape[0] == 1
    img = img[0]
    axis[0].imshow(img.permute(1, 2, 0))
    axis[0].set_title(f"Source ({img.min():g}, {img.max():g})")
    for image, label, ax in zip(simulation, labels, axis[1:]):
        ax.imshow(image[0].cpu().permute(1, 2, 0), vmin=0.0, vmax=1.0)
        ax.set_title(f"{label} simulation ({image.min():g}, {image.max():g})")

    plt.show()
