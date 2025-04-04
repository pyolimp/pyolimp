from __future__ import annotations
from typing import NamedTuple
from pathlib import Path
from pydantic import Field
from collections.abc import Sequence
from .base import StrictModel
from .optimizer import Optimizer, AdamConfig
from .model import Model as ModelConfig
from .dataset import ImgDataloaderConfig, ProgressContext
from .loss_function import LossFunction
from .distortion import DistortionConfig

from torch.utils.data import Dataset
from torch import Tensor
from .....simulate import Distortion


class DistortionsGroup(NamedTuple):
    """
    datasets: variable arguments for distortion
    distortions_classes: distortions

    Examples:
        dataset is psf, distortions_class is `RefractionDistortion`
        dataset is None, distortions_class is `ColorBlindnessDistortion`
    """

    datasets: list[Dataset[Tensor] | None]
    distortions: list[Distortion]

    def create(self, arguments: Sequence[Tensor]):
        outs_len = len([ds for ds in self.datasets if ds is not None])

        def apply_distortion(
            original_image: Tensor,
        ) -> Tensor:
            assert len(arguments) == outs_len, (len(arguments), outs_len)
            dataset_idx = 0
            for distortion, d_input in zip(
                self.distortions, self.datasets, strict=True
            ):
                if d_input is None:  # None = no arguments
                    the_distortion = distortion()
                else:
                    the_distortion = distortion(arguments[dataset_idx])
                    dataset_idx += 1
                original_image = the_distortion(original_image).clip(
                    min=0.0, max=1.0
                )
            return original_image

        return apply_distortion


class Config(StrictModel):
    """
    Root configuration class
    """

    model: ModelConfig
    img: ImgDataloaderConfig
    distortion: list[DistortionConfig]
    random_seed: int = 47
    batch_size: int = 1
    sample_size: int = Field(1000, description="Number of items for one epoch")
    train_frac: float = 0.8
    validation_frac: float = 0.2
    epoch_dir: Path = Field(
        default=Path("./epoch_saved"),
        description="Where to save .pth files",
    )
    optimizer: Optimizer = AdamConfig(name="Adam")
    epochs: int = Field(50, description="Maximal number of epochs to run")
    loss_function: LossFunction
    # stop criterion
    patience: int = Field(
        default=10,
        description="The number of epochs the model "
        "is allowed to go without improving",
    )

    def load_distortions(
        self, progress_context: ProgressContext
    ) -> DistortionsGroup:
        datasets: list[Dataset[Tensor] | None] = []
        distortions_classes: list[Distortion] = []
        for distortion in self.distortion:
            dataset, distortion = distortion.load(progress_context)
            datasets.append(dataset)
            distortions_classes.append(distortion)
        return DistortionsGroup(datasets, distortions_classes)
