from __future__ import annotations
from typing import NamedTuple
from pathlib import Path
from pydantic import Field
from .base import StrictModel
from .optimizer import Optimizer, AdamConfig
from .model import Model as ModelConfig
from .dataset import ImgDataloaderConfig
from .loss_function import LossFunction
from .distortion import DistortionConfig

from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms.v2 import Compose
from .....simulate import Distortion


class DistortionsGroup(NamedTuple):
    datasets: list[Dataset[Tensor]]
    composees: list[Compose]
    distortions_classes: list[type[Distortion]]


class Config(StrictModel):
    model: ModelConfig
    img: ImgDataloaderConfig
    distortion: list[DistortionConfig]
    random_seed: int = 47
    batch_size: int = 1
    sample_size: int = Field(
        default=1000, description="Number of items for one epoch"
    )
    train_frac: float = 0.8
    validation_frac: float = 0.2
    epoch_dir: Path = Field(
        default=Path("./epoch_saved"),
        description="Where to save .pth files",
    )
    optimizer: Optimizer = AdamConfig(name="Adam")
    epochs: int = Field(
        default=50, description="Maximal number of epochs to run"
    )
    loss_function: LossFunction

    def load_distortions(self) -> DistortionsGroup:
        datasets: list[Dataset[Tensor]] = []
        composees: list[Compose] = []
        distortions_classes: list[type[Distortion]] = []
        for distortion in self.distortion:
            dataset, compose, distortion_cls = distortion.load()
            datasets.append(dataset)
            composees.append(compose)
            distortions_classes.append(distortion_cls)
        return DistortionsGroup(datasets, composees, distortions_classes)
