from __future__ import annotations
from typing import Literal, Callable, Iterable
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import Resize

from olimp.simulate import ApplyDistortion
from olimp import demo_data


def demo(
    name: Literal["ColorBlindnessDistortion", "RefractionDistortion"],
    sim_functions: Callable[[], Iterable[tuple[ApplyDistortion, str]]],
    on: Literal["ishihara", "horse"] = "ishihara",
    size: tuple[int, int] = (256, 256),
):
    assert on in ["ishihara", "horse"], on
    img = demo_data.ishihara() if on == "ishihara" else demo_data.horse()
    img = img[None] / 255.0
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
