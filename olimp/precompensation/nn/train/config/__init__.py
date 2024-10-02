from __future__ import annotations
from typing import Literal
from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
from .base import StrictModel
from .transform import (
    Transforms,
    GrayscaleTransform,
    ResizeTransform,
    PSFNormalizeTransform,
    Float32Transform,
)
from .optimizer import Optimizer, AdamConfig
from .model import Model as ModelConfig
from .dataset import Dataset
from .loss_function import LossFunction, VaeLossFunction


class BaseDataloaderConfig(StrictModel):
    transforms: Transforms
    datasets: list[Dataset]

    def load(self):
        all_transforms = [t.transform() for t in self.transforms]
        if not all_transforms:
            all_transforms.append(lambda a: a)
        transforms = Compose(all_transforms)
        dataset = ConcatDataset([dataset.load() for dataset in self.datasets])
        return dataset, transforms


class ImgDataloaderConfig(BaseDataloaderConfig):
    transforms: Transforms = [
        GrayscaleTransform(name="Grayscale"),
        ResizeTransform(name="Resize"),
    ]


class PsfDataloaderConfig(BaseDataloaderConfig):
    transforms: Transforms = [
        Float32Transform(name="Float32"),
        PSFNormalizeTransform(name="PSFNormalize"),
    ]


class Config(StrictModel):
    model: ModelConfig
    img: ImgDataloaderConfig
    psf: PsfDataloaderConfig | None = None
    random_seed: int = 47
    batch_size: int = 1
    train_frac: float = 0.8
    validation_frac: float = 0.2
    epoch_dir: Path = Path("./epoch_saved")
    optimizer: Optimizer = AdamConfig(name="Adam")
    epochs: int = 50
    loss_function: LossFunction = VaeLossFunction(name="Vae")
