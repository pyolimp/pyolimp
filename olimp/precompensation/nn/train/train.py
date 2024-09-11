from __future__ import annotations
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

c = Progress()
c.start()
load_task = c.add_task("Import libraries")
from typing import Literal, Annotated, TYPE_CHECKING
from math import prod

import random
from pydantic import BaseModel, Field
import json5

c.update(load_task, completed=1)
import torch

c.update(load_task, completed=50)
from olimp.processing import conv
from torch import nn, Tensor, tensor
from torchvision.transforms.v2 import Compose, Resize, Grayscale

c.update(load_task, completed=75)
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

c.update(load_task, completed=100)


#
# ModelConfig
#


class ModelConfig(BaseModel):
    pass


class VDSR(ModelConfig):
    name: Literal["vdsr"]

    def get_instance(self):
        from ..models.vdsr import VDSR

        return VDSR()


class VAE(ModelConfig):
    name: Literal["vae"]

    def get_instance(self):
        from ..models.vae import VAE

        return VAE()


class PrecompensationUSRNet(ModelConfig):
    name: Literal["precompensationusrnet"]

    def get_instance(self):
        from ..models.usrnet import PrecompensationUSRNet

        return PrecompensationUSRNet()


class PrecompensationDWDN(ModelConfig):
    name: Literal["precompensationdwdn"]
    n_levels: int = 1

    def get_instance(self):
        from ..models.dwdn import PrecompensationDWDN

        return PrecompensationDWDN(n_levels=self.n_levels)


#
# DataLoaderConfig
#


class DataLoaderConfig(BaseModel):
    pass


class SCA2023(DataLoaderConfig):
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

    def load(self):
        from ..dataloader.sca_2023 import SCA2023Dataset

        return SCA2023Dataset(self.subsets)


#
# Transform
#


class TransformsTransform(BaseModel):
    pass


class ResizeTransform(TransformsTransform):
    name: Literal["Resize"]
    size: list[int] = [512, 512]

    def transform(self):
        return Resize(self.size)


class GrayscaleTransform(TransformsTransform):
    name: Literal["Grayscale"]

    def transform(self):
        return Grayscale()


#
#
#

OlimpDataset = Annotated[SCA2023, Field(..., discriminator="name")]
Transforms = Annotated[
    list[ResizeTransform | GrayscaleTransform], Field(discriminator="name")
]


class DataloaderConfig(BaseModel):
    transforms: Transforms
    dataset: OlimpDataset

    def load(self):
        dataset = self.dataset.load()
        transforms = Compose([t.transform() for t in self.transforms])

        return dataset, transforms


class ImgDataloaderConfig(DataloaderConfig):

    transforms: Transforms = Field(
        discriminator="name",
        default=[
            GrayscaleTransform(name="Grayscale"),
            ResizeTransform(name="Resize"),
        ],
    )


class PsfDataloaderConfig(DataloaderConfig):
    transforms: Transforms = Field(
        discriminator="name",
        default=[
            ResizeTransform(name="Resize"),
            GrayscaleTransform(name="Grayscale"),
        ],
    )


class Config(BaseModel):
    model: VDSR | VAE = Field(..., discriminator="name")
    img: ImgDataloaderConfig
    psf: PsfDataloaderConfig | None
    random_seed: int = 47
    batch_size: int = 1
    train_frac: float = 0.8
    validation_frac: float = 0.2
    epoch_path: Path = Path("./epoch_saved")


from ....evaluation.loss import ms_ssim


class ShuffeledDataset(Dataset[Tensor]):
    def __init__(self, dataset: Dataset[Tensor], indices: list[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, index: int) -> Tensor:
        assert 0 <= index < len(self._indices)
        index = self._indices[index]
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._indices)


class ProductDataset(Dataset[tuple[Tensor, ...]]):
    datasets: tuple[Dataset[Tensor], ...]

    def __init__(self, *datasets: Dataset[Tensor]) -> None:
        self._datasets = datasets
        self._sizes = [len(dataset) for dataset in datasets]

    def __getitem__(self, index: int):
        out: list[Tensor] = []
        for dataset, size in zip(self._datasets, self._sizes):
            index, cur = divmod(index, size)
            out.append(dataset[cur])
        return tuple(out)

    def __len__(self):
        return prod(self._sizes)


def random_split(
    dataset: Dataset[Tensor], train_frac: float, validation_frac: float
) -> tuple[ShuffeledDataset, ShuffeledDataset, ShuffeledDataset]:
    size = len(dataset)
    train_size = round(train_frac * size)
    val_size = round(validation_frac * size)
    test_size = size - (train_size + val_size)

    indices = torch.randperm(len(dataset), device="cpu")

    return (
        ShuffeledDataset(dataset, indices[:train_size].tolist()),
        ShuffeledDataset(
            dataset, indices[train_size : train_size + test_size].tolist()
        ),
        ShuffeledDataset(dataset, indices[train_size + test_size :].tolist()),
    )


def custom_collate_fn(batch):
    breakpoint()
    psfs, images, original_sums = zip(*batch)

    # Repeat the first PSF in the batch across the batch dimension
    psfs = torch.stack(psfs)
    psf = psfs[0].unsqueeze(0).repeat(len(images), 1, 1, 1)

    images = torch.stack(images)
    original_sums = torch.tensor(original_sums)

    return psf, images, original_sums


def _evaluate_dataset(model: nn.Module, dl: DataLoader[tuple[Tensor, ...]]):
    model_kwargs = {}

    for train_data in dl:
        image, psf = train_data
        image = tensor(image)
        psf = tensor(psf)
        image = img_transform(image)
        psf = psf_transform(psf)
        image = image.to(torch.float32) / 255.0
        psf /= psf.sum(axis=(1, 2, 3)).view(-1, 1, 1, 1)
        inputs = model.preprocess(image, psf)

        precompensated = model(
            inputs,
            **model.arguments(inputs, psf, **model_kwargs),
        )

        loss_func = ms_ssim
        retinal_precompensated = conv(
            precompensated.to(torch.float32).clip(0, 1), psf
        )
        loss = loss_func(retinal_precompensated, image)
        yield loss


def should_stop_early(train_loss: list[float], val_loss: list[float]) -> bool:
    return False


def main(
    model: nn.Module,
    img_dataset: Dataset[Tensor],
    img_transform: Compose,
    psf_dataset: Dataset[Tensor],
    psf_transform: Compose,
    random_seed: int,
    batch_size: int,
    train_frac: float,
    validation_frac: float,
    epochs: int,
    epoch_path: Path,
):
    epoch_path.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    img_dataset_train, img_dataset_validation, img_dataset_test = random_split(
        img_dataset, train_frac, validation_frac
    )

    psf_dataset_train, psf_dataset_validation, psf_dataset_test = random_split(
        psf_dataset, train_frac, validation_frac
    )

    dataset_train = ProductDataset(img_dataset_train, psf_dataset_train)
    dataset_validation = ProductDataset(
        img_dataset_validation, psf_dataset_validation
    )
    dataset_test = ProductDataset(img_dataset_test, psf_dataset_test)

    dl_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

    if dataset_validation:
        dl_validation = DataLoader(
            dataset_validation, shuffle=True, batch_size=batch_size
        )
    else:
        dl_validation = None

    if dataset_test:
        dl_test = DataLoader(
            dataset_test, shuffle=False, batch_size=batch_size
        )
    else:
        dl_test = None

    initial_lr = 0.00001
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr
    )  # torch.optim.SGD

    model_name = type(model).__name__

    # device = next(model.parameters()).device

    global_train_loss: list[float] = []
    global_val_loss: list[float] = []

    best_train_loss = float("inf")
    best_val_loss = float("inf")

    c.stop()

    p = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )
    p.start()

    for epoch in p.track(range(epochs), total=epochs, description="Epoch..."):
        model.train()

        # training
        train_loss = 0.0
        for loss in p.track(
            _evaluate_dataset(model, dl_train),
            total=len(dl_train),
            description="Training...",
        ):
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(dl_train)
        global_train_loss.append(train_loss)

        # validation
        model.eval()
        if dl_validation is not None:
            val_loss = 0.0

            for loss in p.track(
                _evaluate_dataset(model, dl_validation),
                total=len(dl_validation),
                description="Validation...",
            ):
                val_loss += loss.item()
            val_loss /= len(dl_validationval)
            global_val_loss.append(val_loss)

        # save
        torch.save(
            model.state_dict(), epoch_path / f"{model_name}_{epoch}.pth"
        )

        if should_stop_early(global_train_loss, global_val_loss):
            break

    # test
    if dl_test is not None:
        test_loss = 0.0
        for loss in p.track(
            _evaluate_dataset(model, dl_test),
            total=len(dl_test),
            description="Test...",
        ):
            test_loss += loss.item()


if __name__ == "__main__":

    with open("./schema.json", "w") as out:
        schema = Config.schema_json()
        out.write(schema)
    with open("./config.json") as f:
        data = json5.load(f)
    config = Config(**data)

    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = config.model.get_instance()
        img_dataset, img_transform = config.img.load()
        psf_dataset, psf_transform = config.psf.load()
        main(
            model,
            img_dataset,
            img_transform,
            psf_dataset,
            psf_transform,
            random_seed=config.random_seed,
            batch_size=config.batch_size,
            train_frac=config.train_frac,
            validation_frac=config.validation_frac,
            epochs=42,
            epoch_path=config.epoch_path,
        )
