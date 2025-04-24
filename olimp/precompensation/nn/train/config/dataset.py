from __future__ import annotations
from typing import Annotated, Literal
from random import Random
from pydantic import Field
from pathlib import Path
from .base import StrictModel

from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from torch import Tensor
import torch
from .transform import BallfishTransforms
from ballfish import create_augmentation, Datum
from ballfish.distribution import DistributionParams
from ...dataset import ProgressContext


class DatasetConfig(StrictModel):
    limit: int | None = Field(
        default=None, description="Load dataset, but only take first N images"
    )

    def load(self, progress_context: ProgressContext) -> TorchDataset[Tensor]:
        raise NotImplementedError


class SCA2023(DatasetConfig):
    name: Literal["SCA2023"]

    subsets: set[
        Literal[
            "Images",
            "Images/Icons",
            "Images/Real_images/Animals",
            "Images/Real_images",
            "Images/Real_images/Faces",
            "Images/Real_images/Natural",
            "Images/Real_images/Urban",
            "Images/Texts",
            "PSFs",
            "PSFs/Broad",
            "PSFs/Medium",
            "PSFs/Narrow",
        ]
    ]

    def load(self, progress_context: ProgressContext):
        from ...dataset.sca_2023 import SCA2023Dataset

        return SCA2023Dataset(
            self.subsets, limit=self.limit, progress_context=progress_context
        )


class Olimp(DatasetConfig):
    name: Literal["Olimp"]

    subsets: set[
        Literal[
            "*",  # load all
            "abstracts and textures",
            "abstracts and textures/abstract art",
            "abstracts and textures/backgrounds and patterns",
            "abstracts and textures/colorful abstracts",
            "abstracts and textures/geometric shapes",
            "abstracts and textures/neon abstracts",
            "abstracts and textures/textures",
            "animals",
            "animals/birds",
            "animals/farm animals",
            "animals/insects and spiders",
            "animals/marine life",
            "animals/pets",
            "animals/wild animals",
            "art and culture",
            "art and culture/cartoon and comics",
            "art and culture/crafts and handicrafts",
            "art and culture/dance and theater performances",
            "art and culture/music concerts and instruments",
            "art and culture/painting and frescoes",
            "art and culture/sculpture and bas-reliefs",
            "food and drinks",
            "food and drinks/desserts and bakery",
            "food and drinks/dishes",
            "food and drinks/drinks",
            "food and drinks/food products on store shelves",
            "food and drinks/fruits and vegetables",
            "food and drinks/street food",
            "interiors",
            "interiors/gyms and pools",
            "interiors/living spaces",
            "interiors/museums and galleries",
            "interiors/offices",
            "interiors/restaurants and cafes",
            "interiors/shopping centers and stores",
            "nature",
            "nature/beaches",
            "nature/deserts",
            "nature/fields and meadows",
            "nature/forest",
            "nature/mountains",
            "nature/water bodies",
            "objects and items",
            "objects and items/books and stationery",
            "objects and items/clothing and accessories",
            "objects and items/electronics and gadgets",
            "objects and items/furniture and decor",
            "objects and items/tools and equipment",
            "objects and items/toys and games",
            "portraits and people",
            "portraits and people/athletes and dancers",
            "portraits and people/crowds and demonstrations",
            "portraits and people/group photos",
            "portraits and people/individual portraits",
            "portraits and people/models on runway",
            "portraits and people/workers in their workplaces",
            "sports and active leisure",
            "sports and active leisure/cycling and rollerblading",
            "sports and active leisure/extreme sports",
            "sports and active leisure/individual sports",
            "sports and active leisure/martial arts",
            "sports and active leisure/team sports",
            "sports and active leisure/tourism and hikes",
            "text and pictogram",
            "text and pictogram/billboard text",
            "text and pictogram/blueprints",
            "text and pictogram/caricatures and pencil drawing",
            "text and pictogram/text documents",
            "text and pictogram/traffic signs",
            "urban scenes",
            "urban scenes/architecture",
            "urban scenes/city at night",
            "urban scenes/graffiti and street art",
            "urban scenes/parks and squares",
            "urban scenes/streets and avenues",
            "urban scenes/transport",
        ]
    ]

    def load(self, progress_context: ProgressContext):
        from ...dataset.olimp import OlimpDataset

        return OlimpDataset(
            self.subsets, limit=self.limit, progress_context=progress_context
        )


class Directory(DatasetConfig):
    name: Literal["Directory"]
    path: Path
    matches: list[str] = ["*.jpg", "*.jpeg", "*.png"]

    def load(self, progress_context: ProgressContext):
        from ...dataset.directory import DirectoryDataset

        return DirectoryDataset(self.path, self.matches, limit=self.limit)


class CVD(DatasetConfig):
    name: Literal["CVD"]
    subsets: set[
        Literal[
            "Color_cvd_D_experiment_100000",
            "Color_cvd_P_experiment_100000",
            "*",
        ]
    ]

    def load(self, progress_context: ProgressContext):
        from ...dataset.cvd import CVDDataset

        return CVDDataset(
            self.subsets, limit=self.limit, progress_context=progress_context
        )


class PSFGauss(DatasetConfig):
    name: Literal["psf_gauss"]
    width: int = 512
    height: int = 512
    center_x: DistributionParams | None = None
    center_y: DistributionParams | None = None
    theta: DistributionParams = 0.0
    sigma_x: DistributionParams = 5.0
    sigma_y: DistributionParams = 5.0
    seed: int = 42
    size: int = 10000

    def load(self, progress_context: ProgressContext):
        from ...dataset.psf_gauss import PsfGaussDataset

        x = self.width * 0.5 if self.center_x is None else self.center_x
        y = self.height * 0.5 if self.center_y is None else self.center_y
        return PsfGaussDataset(
            width=self.width,
            height=self.height,
            center_x=x,
            center_y=y,
            theta=self.theta,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            seed=self.seed,
            size=self.size,
        )


class PSFSCA(DatasetConfig):
    name: Literal["psf_sca"]
    width: int = 512
    height: int = 512
    sphere_dpt: DistributionParams = -1.0
    cylinder_dpt: DistributionParams = 0.0
    angle_deg: DistributionParams = 0.0
    pupil_diameter_mm: DistributionParams = 4.0
    am2px: float = 0.001
    seed: int = 42
    size: int = 10000

    def load(self, progress_context: ProgressContext):
        from ...dataset.psf_sca import PSFSCADataset

        return PSFSCADataset(
            width=self.width,
            height=self.height,
            sphere_dpt=self.sphere_dpt,
            cylinder_dpt=self.cylinder_dpt,
            angle_deg=self.angle_deg,
            pupil_diameter_mm=self.pupil_diameter_mm,
            am2px=self.am2px,
            seed=self.seed,
            size=self.size,
        )


Dataset = Annotated[
    SCA2023 | Olimp | CVD | Directory | PSFGauss | PSFSCA,
    Field(..., discriminator="name"),
]


class BaseDataloaderConfig(StrictModel):
    transforms: BallfishTransforms | None
    datasets: list[Dataset]
    augmentation_factor: int = Field(
        default=1,
        description="When transformations augment the dataset, it is helpful "
        "to set augmentation_factor that will multiple original dataset size",
    )

    def load(self, progress_context: ProgressContext):
        dataset = ConcatDataset[Tensor](
            [dataset.load(progress_context) for dataset in self.datasets]
        )
        if self.transforms:
            dataset = AugmentedDataset(
                dataset, self.transforms, self.augmentation_factor
            )

        return dataset


class AugmentedDataset(TorchDataset[Tensor]):
    def __init__(
        self,
        inner_dataset: TorchDataset[Tensor],
        augmentation: BallfishTransforms,
        augmentation_factor: int = 1,
    ) -> None:
        self._inner_dataset = inner_dataset
        if not any(item["name"] == "rasterize" for item in augmentation):
            augmentation.insert(0, {"name": "_copy"})
        self._augmentation = create_augmentation(augmentation)
        self._random = Random(b"helloseed")
        self._inner_dataset_size = len(inner_dataset)
        self._size = self._inner_dataset_size * augmentation_factor

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Tensor:
        tensor = self._inner_dataset[index % self._inner_dataset_size]
        with torch.device("cpu"):
            datum = self._augmentation(
                Datum(source=tensor.unsqueeze(0).to(dtype=torch.float32)),
                self._random,
            )
            assert datum.image is not None
        return datum.image.squeeze(0)


class ImgDataloaderConfig(BaseDataloaderConfig):
    transforms: BallfishTransforms | None = [
        {"name": "grayscale"},
        {"name": "resize", "width": 512, "height": 512},
        # {"name": "float32"}, converted to f32 when passed to `Datum``
        {"name": "divide", "value": 255.0},
    ]


class PsfDataloaderConfig(BaseDataloaderConfig):
    transforms: BallfishTransforms | None = [
        # {"name": "float32"}, converted to f32 when passed to `Datum`
        {"name": "psf_normalize"},
    ]
