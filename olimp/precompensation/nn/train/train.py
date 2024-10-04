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

# import warnings

# warnings.filterwarnings("error", category=UserWarning)

load_task = c.add_task("Import libraries")
from typing import Any, Callable, TypeAlias, Annotated
from math import prod

import random
import json5

c.update(load_task, completed=1)
import torch

c.update(load_task, completed=50)
from torch import nn, Tensor
from torchvision.transforms.v2 import Compose

c.update(load_task, completed=75)
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

c.update(load_task, completed=100)
from .config import Config

# from ....evaluation.loss import ms_ssim, vae_loss

LossFunction: TypeAlias = Annotated[
    Callable[[list[Tensor], list[Tensor]], Tensor],
    "(model result, model input before preprocessing) -> loss",
]


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
    transforms: tuple[Compose, ...],
    loss_function: LossFunction,
):
    model_kwargs = {}
    device = next(model.parameters()).device

    for train_data in dl:
        datums: list[Tensor] = []
        for datum, transform in zip(train_data, transforms, strict=True):
            datum = datum.to(device)
            datums.append(transform(datum))
        inputs = model.preprocess(*datums)

        precompensated = model(
            inputs,
            **model.arguments(inputs, datums[-1], **model_kwargs),
        )
        assert isinstance(precompensated, tuple | list), (
            f"All models MUST return tuple "
            f"({model} returned {type(precompensated)})"
        )
        loss = loss_function(precompensated, datums)
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
    transforms: tuple[Compose, ...],
    loss_function: LossFunction,
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
                transforms=transforms,
                loss_function=loss_function,
            ),
            task_id=training_task,
        ):
            train_loss += loss.item()
            p.update(training_task, loss=f"{loss.item():g}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p.remove_task(training_task)
        assert train_loss, train_loss

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
                    transforms=transforms,
                    loss_func=loss_func,
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
            p.console.log("Stop early")
            break


def train(
    model: nn.Module,
    img_dataset: Dataset[Tensor],
    img_transform: Compose,
    psf_dataset: Dataset[Tensor] | None,
    psf_transform: Compose | None,
    random_seed: int,
    batch_size: int,
    train_frac: float,
    validation_frac: float,
    epochs: int,
    epoch_dir: Path,
    create_optimizer: Callable[[nn.Module], torch.optim.optimizer.Optimizer],
    loss_function: LossFunction,
):
    epoch_dir.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    img_dataset_train, img_dataset_validation, img_dataset_test = random_split(
        img_dataset, train_frac, validation_frac
    )

    if psf_dataset is not None:
        psf_dataset_train, psf_dataset_validation, psf_dataset_test = (
            random_split(psf_dataset, train_frac, validation_frac)
        )
        dataset_train = ProductDataset(img_dataset_train, psf_dataset_train)
        dataset_validation = ProductDataset(
            img_dataset_validation, psf_dataset_validation
        )
        dataset_test = ProductDataset(img_dataset_test, psf_dataset_test)
        assert psf_transform is not None
        transforms = img_transform, psf_transform
    else:
        dataset_train = ProductDataset(img_dataset_train)
        c.console.log(f"Train: {len(dataset_train)} items")
        dataset_validation = ProductDataset(img_dataset_validation)
        c.console.log(f"Validation: {len(dataset_validation)} items")
        dataset_test = ProductDataset(img_dataset_test)
        c.console.log(f"Test: {len(dataset_test)} items")
        transforms = (img_transform,)

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
            transforms=transforms,
            loss_function=loss_function,
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
            _evaluate_dataset(
                model,
                dl_test,
                transforms=transforms,
                loss_function=loss_function,
            ),
            task_id=test_task,
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
        c.console.log(f"[green] {schema_path} [cyan]saved")
        c.stop()
        return
    c.console.log(f"Using [green]{args.config}")

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
        loss_function = config.loss_function.load(model)
        img_dataset, img_transform = config.img.load()
        # img_dataset._items = img_dataset._items[0:10]
        if config.psf is not None:
            psf_dataset, psf_transform = config.psf.load()
        else:
            psf_dataset = psf_transform = None
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
            loss_function=loss_function,
        )


if __name__ == "__main__":
    main()
