from __future__ import annotations
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskID,
)

c = Progress()
c.start()

import warnings

warnings.filterwarnings("error", category=UserWarning)

load_task = c.add_task("Import libraries")
from typing import Literal, Annotated, Callable
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


class BaseModel(BaseModel):
    class Config:
        extra = "forbid"


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
    
    
class UNET_b0(ModelConfig):
    name: Literal["unet_b0"]

    def get_instance(self):
        from ..models.unet_efficient_b0 import PrecompensationUNETB0

        return PrecompensationUNETB0()


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
# DatasetrConfig
#


class DatasetrConfig(BaseModel):
    pass


class SCA2023(DatasetrConfig):
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
        from ..dataset.sca_2023 import SCA2023Dataset

        return SCA2023Dataset(self.subsets)


class Olimp(DatasetrConfig):
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

    def load(self):
        from ..dataset.olimp import OlimpDataset

        return OlimpDataset(self.subsets)


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
    num_output_channels: int = 1

    def transform(self):
        return Grayscale(self.num_output_channels)


#
# Optimizers
#


class AdamConfig(BaseModel):
    name: Literal["Adam"]
    learning_rate: float = 0.00001
    eps: float = 1e-8

    def load(self):
        def optimizer(model: torch.nn.Module):
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return optimizer


class SGDConfig(BaseModel):
    name: Literal["SGD"]
    learning_rate: float = 0.00001

    def load(self):
        def optimizer(model: torch.nn.Module):
            return torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        return optimizer


#
#
#

OlimpDataset = Annotated[SCA2023, Field(..., discriminator="name")]
Optimizer = Annotated[AdamConfig | SGDConfig, Field(..., discriminator="name")]
Transforms = list[
    Annotated[
        ResizeTransform | GrayscaleTransform, Field(discriminator="name")
    ]
]


class DataloaderConfig(BaseModel):
    transforms: Transforms
    dataset: OlimpDataset

    def load(self):
        dataset = self.dataset.load()
        all_transforms = [t.transform() for t in self.transforms]
        if not all_transforms:
            all_transforms.append(lambda a: a)
        transforms = Compose(all_transforms)

        return dataset, transforms


class ImgDataloaderConfig(DataloaderConfig):

    transforms: Transforms = [
        GrayscaleTransform(name="Grayscale"),
        ResizeTransform(name="Resize"),
    ]


class PsfDataloaderConfig(DataloaderConfig):
    transforms: Transforms = [
        # ResizeTransform(name="Resize"),
        # GrayscaleTransform(name="Grayscale"),
    ]


class Config(BaseModel):
    model: VDSR | VAE | UNET_b0 | PrecompensationUSRNet | PrecompensationDWDN = Field(..., discriminator="name")
    img: ImgDataloaderConfig
    psf: PsfDataloaderConfig | None
    random_seed: int = 47
    batch_size: int = 1
    train_frac: float = 0.8
    validation_frac: float = 0.2
    epoch_dir: Path = Path("./epoch_saved")
    optimizer: Optimizer = AdamConfig(name="Adam")
    epochs: int = 50


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


def _evaluate_dataset(
    model: nn.Module,
    dl: DataLoader[tuple[Tensor, ...]],
    img_transform: Compose,
    psf_transform: Compose,
):
    model_kwargs = {}
    device = next(model.parameters()).device
    print("!!! A", device)

    for train_data in dl:
        image, psf = train_data
        image = image.to(device)
        psf = psf.to(device)
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


class TrainStatistics:
    def __init__(self, patience: int = 10):
        self.train_loss: list[float] = []
        self.validation_loss: list[float] = []
        self.best_train_loss = float("inf")
        self.best_validation_loss = float("inf")
        self.epochs_no_improve = 0

        self.is_best_train = True
        self.is_best_validation = True
        self._patience = patience

    def __call__(self, train_loss: float, validation_loss: float):
        self.is_best_validation = validation_loss < self.best_validation_loss
        if self.is_best_validation:
            self.best_validation_loss = validation_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        self.is_best_train = train_loss < self.best_train_loss
        if self.is_best_train:
            self.best_train_loss = train_loss

        self.train_loss

    def should_stop_early(self) -> bool:
        return self.epochs_no_improve >= self._patience


def _train_loop(
    p: Progress,
    model: torch.nn.Module,
    epochs: int,
    dl_train: DataLoader[tuple[Tensor, ...]],
    dl_validation: DataLoader[tuple[Tensor, ...]] | None,
    epoch_task: TaskID,
    optimizer: torch.optim.optimizer.Optimizer,
    epoch_dir: Path,
    img_transform: Compose,
    psf_transform: Compose,
):
    train_statistics = TrainStatistics(patience=3)
    model_name = type(model).__name__

    for epoch in p.track(range(epochs), task_id=epoch_task):
        model.train()

        training_task = p.add_task(
            "Training...", total=len(dl_train), loss="?"
        )

        # training
        train_loss = 0.0
        for loss in p.track(
            _evaluate_dataset(
                model,
                dl_train,
                img_transform=img_transform,
                psf_transform=psf_transform,
            ),
            task_id=training_task,
        ):
            train_loss += loss.item()
            p.update(training_task, loss=f"{loss.item():g}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p.remove_task(training_task)

        train_loss /= len(dl_train)

        p.update(epoch_task, loss=f"{train_loss:g}")

        # validation
        model.eval()
        validation_loss = 0.0
        if dl_validation is not None:
            for loss in p.track(
                _evaluate_dataset(
                    model,
                    dl_validation,
                    img_transform=img_transform,
                    psf_transform=psf_transform,
                ),
                total=len(dl_validation),
                description="Validation...",
            ):
                validation_loss += loss.item()
            validation_loss /= len(dl_validation)

        train_statistics(train_loss, validation_loss)

        # save
        cur_epoch_path = epoch_dir / f"{model_name}_{epoch:04d}.pth"
        torch.save(model.state_dict(), cur_epoch_path)

        if train_statistics.is_best_train:
            best_train_path = cur_epoch_path.with_name("best_train.pth")
            best_train_path.unlink(missing_ok=True)
            best_train_path.hardlink_to(cur_epoch_path)

        if train_statistics.is_best_validation:
            best_validation_path = cur_epoch_path.with_name(
                "best_validation.pth"
            )
            best_validation_path.unlink(missing_ok=True)
            best_validation_path.hardlink_to(cur_epoch_path)

        if train_statistics.should_stop_early():
            p.console.print("Stop early")
            break


def train(
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
    epoch_dir: Path,
    create_optimizer: Callable[[nn.Module], torch.optim.optimizer.Optimizer],
):
    epoch_dir.mkdir(exist_ok=True, parents=True)
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

    optimizer = create_optimizer(model)
    c.stop()

    p = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn("loss: {task.fields[loss]}"),
    )
    p.start()
    epoch_task = p.add_task("Epoch...", total=epochs, loss="?")

    try:
        _train_loop(
            p,
            model=model,
            epochs=epochs,
            dl_train=dl_train,
            dl_validation=dl_validation,
            epoch_task=epoch_task,
            optimizer=optimizer,
            epoch_dir=epoch_dir,
            img_transform=img_transform,
            psf_transform=psf_transform,
        )
    except KeyboardInterrupt:
        p.print("training stopped by user (Ctrl+C)")

    p.console.print(p)
    p.remove_task(epoch_task)

    # test
    if dl_test is not None:
        test_task = p.add_task("Test... ", total=len(dl_test), loss="?")
        test_loss = 0.0
        for loss in p.track(
            _evaluate_dataset(model, dl_test), task_id=test_task
        ):
            test_loss += loss.item()
            p.update(test_task, loss=f"{test_loss:g}")
        test_loss /= len(dl_test)
        p.update(test_task, loss=f"{test_loss:g}")
        p.console.print(p, end="")
        p.remove_task(test_task)
    p.stop()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="./olimp/precompensation/nn/pipeline/vdsr.json",
    )
    parser.add_argument(
        "--update-schema",
        action="store_true",
    )
    args = parser.parse_args()
    
    if args.update_schema:
        schema_path = Path(__file__).with_name("schema.json")
        import json

        schema_path.write_text(
            json.dumps(Config.model_json_schema(), ensure_ascii=False)
        )
        return
    
    with args.config.open() as f:
        data = json5.load(f)
    config = Config(**data)

    if torch.cuda.is_available():
        device_str = "cuda"
        c.console.print("Current device: [bold green] GPU")
    else:
        device_str = "cpu"
        c.console.print("Current device: [bold red] CPU")

    with torch.device(device_str):
        model = config.model.get_instance()
        img_dataset, img_transform = config.img.load()
        # img_dataset._items = img_dataset._items[0:10]
        psf_dataset, psf_transform = config.psf.load()
        # psf_dataset._items = psf_dataset._items[0:3]
        create_optimizer = config.optimizer.load()
        train(
            model,
            img_dataset,
            img_transform,
            psf_dataset,
            psf_transform,
            random_seed=config.random_seed,
            batch_size=config.batch_size,
            train_frac=config.train_frac,
            validation_frac=config.validation_frac,
            epochs=config.epochs,
            epoch_dir=config.epoch_dir,
            create_optimizer=create_optimizer,
        )


if __name__ == "__main__":
    main()
