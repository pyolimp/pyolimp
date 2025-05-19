from __future__ import annotations
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskID,
)

with Progress() as ci:
    load_task = ci.add_task("Import libraries")
    from typing import Callable, Any, Protocol, Sequence
    from math import prod

    import random
    import json5
    from contextlib import contextmanager

    ci.update(load_task, completed=1)
    import torch

    ci.update(load_task, completed=50)
    from torch import nn, Tensor

    ci.update(load_task, completed=75)
    from torch.utils.data import Dataset, DataLoader
    from .torch_issue_135990 import RandomSamplerCPU as RandomSampler
    from .log_table import LogTable
    from pathlib import Path

    ci.update(load_task, completed=95)
    from .config import Config, DistortionsGroup, Distortion

    ci.update(load_task, completed=100)


class LossFunction(Protocol):
    def __call__(
        self,
        precompensated: Tensor,
        original_image: Tensor,
        distortion_fn: Callable[[Tensor], Tensor],
        extra: Sequence[Any],
    ) -> Tensor:
        """
        extra - extra arguments for loss function, for example, for VAE loss
        """
        ...


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


def _split_numbers(
    dataset_size: int, train_frac: float, validation_frac: float
) -> tuple[int, int, int]:
    train_size = round(train_frac * dataset_size)
    val_size = round(validation_frac * dataset_size)
    test_size = dataset_size - (train_size + val_size)
    return train_size, val_size, test_size


def random_split(
    dataset: Dataset[Tensor], train_frac: float, validation_frac: float
) -> tuple[ShuffeledDataset, ShuffeledDataset, ShuffeledDataset]:
    train_size, val_size, _test_size = _split_numbers(
        len(dataset), train_frac, validation_frac
    )

    indices = torch.randperm(len(dataset), device="cpu")

    return (
        ShuffeledDataset(dataset, indices[:train_size].tolist()),
        ShuffeledDataset(
            dataset, indices[train_size : train_size + val_size].tolist()
        ),
        ShuffeledDataset(dataset, indices[train_size + val_size :].tolist()),
    )


def _evaluate_dataset(
    model: nn.Module,
    dls: list[DataLoader[Tensor]],
    distortions_group: DistortionsGroup,
    loss_function: LossFunction,
):
    model_kwargs = {}
    device = next(model.parameters()).device

    for batches in zip(*dls, strict=True):
        original_tensors: list[Tensor] = []
        for batch in batches:
            batch = batch.to(device)
            original_tensors.append(batch)

        model_inputs = model.preprocess(*original_tensors)

        precompensated = model(
            model_inputs,
            **model.arguments(model_inputs, original_tensors, **model_kwargs),
        )
        assert isinstance(precompensated, tuple | list), (
            f"All models MUST return tuple "
            f"({model} returned {type(precompensated)})"
        )

        precompensated = model.postprocess(precompensated)
        original_image = original_tensors[0]
        distortion_fn = distortions_group.create(original_tensors[1:])

        loss = loss_function(
            precompensated[0],
            original_image,
            distortion_fn,
            extra=precompensated[1:],
        )
        yield loss


class TrainStatistics:
    def __init__(self, patience: int):
        self.train_loss: list[float] = []
        self.validation_loss: list[float] = []
        self.best_train_loss = float("inf")
        self.best_validation_loss = float("inf")
        self.epochs_no_improve = 0

        self.is_best_train = True
        self.is_best_validation = True
        self._patience = patience

    def __call__(self, train_loss: float, validation_loss: float | None):
        is_best = self.is_best_train = train_loss < self.best_train_loss
        if self.is_best_train:
            self.best_train_loss = train_loss

        if validation_loss is not None:
            is_best = self.is_best_validation = (
                validation_loss < self.best_validation_loss
            )
            if self.is_best_validation:
                self.best_validation_loss = validation_loss
        else:
            self.is_best_validation = False

        if is_best:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

    def should_stop_early(self) -> str | None:
        if self.epochs_no_improve >= self._patience:
            return f"no improvements for {self.epochs_no_improve} epochs"


def _train_loop(
    p: Progress,
    model: torch.nn.Module,
    epochs: int,
    dls_train: list[DataLoader[Tensor]],
    dls_validation: list[DataLoader[Tensor]] | None,
    distortions_group: DistortionsGroup,
    epoch_task: TaskID,
    optimizer: torch.optim.Optimizer,
    epoch_dir: Path,
    loss_function: LossFunction,
    patience: int,
) -> None:
    train_statistics = TrainStatistics(patience=patience)
    model_name = type(model).__name__
    header = [("#", len(str(epochs)) + 2), ("Train", 20)]
    if dls_validation is not None:
        header.append(("Validation", 20))
    log_table = LogTable(header)
    p.console.log(log_table.header())

    for epoch in p.track(range(epochs), task_id=epoch_task):
        model.train()

        training_task = p.add_task(
            "Training...", total=len(dls_train[0]), loss="?"
        )

        # training
        train_loss = 0.0
        for loss in p.track(
            _evaluate_dataset(
                model,
                dls=dls_train,
                distortions_group=distortions_group,
                loss_function=loss_function,
            ),
            task_id=training_task,
        ):
            train_loss += loss.item()
            p.update(training_task, loss=f"{loss.item():g}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        assert train_loss, train_loss

        train_loss /= len(dls_train[0])

        p.update(epoch_task, loss=f"{train_loss:g}")

        # validation
        model.eval()
        validation_loss: float | None = None
        if dls_validation is not None:
            validating_task = p.add_task(
                "Validating...", total=len(dls_validation[0]), loss="?"
            )
            validation_loss = 0.0
            with torch.inference_mode():
                for loss in p.track(
                    _evaluate_dataset(
                        model,
                        dls=dls_validation,
                        distortions_group=distortions_group,
                        loss_function=loss_function,
                    ),
                    task_id=validating_task,
                ):
                    validation_loss += loss.item()
                validation_loss /= len(dls_validation[0])
                p.remove_task(validating_task)
            p.console.log(log_table.row([epoch, train_loss, validation_loss]))
        else:
            p.console.log(log_table.row([epoch, train_loss]))

        p.remove_task(training_task)

        train_statistics(train_loss, validation_loss)

        # save
        cur_epoch_path = epoch_dir / f"{model_name}_{epoch:04d}.pth"
        torch.save(model.state_dict(), cur_epoch_path)

        if train_statistics.is_best_train:
            best_train_path = cur_epoch_path.with_name(
                f"{model_name}_best_train.pth"
            )
            best_train_path.unlink(missing_ok=True)
            best_train_path.hardlink_to(cur_epoch_path)

        if train_statistics.is_best_validation:
            best_validation_path = cur_epoch_path.with_name(
                f"{model_name}_best_validation.pth"
            )
            best_validation_path.unlink(missing_ok=True)
            best_validation_path.hardlink_to(cur_epoch_path)

        if reason := train_statistics.should_stop_early():
            p.console.log(log_table.end())
            p.console.log(f"Stop early: {reason}")
            return
    p.console.log(log_table.end())


def _as_dataloaders(
    datasets: list[Dataset[Tensor]],
    batch_size: int,
    sample_size: int,
) -> list[DataLoader[Tensor]] | None:
    dataloaders: list[DataLoader[Tensor]] = []
    if not sample_size:
        return None
    if any(len(dataset) == 0 for dataset in datasets):
        ci.log("[red] one of the datasets is empty. that is unexpected")
        return None
    for dataset in datasets:
        sampler = RandomSampler(dataset, num_samples=sample_size)
        # ci.log(f"Created sampler with {sample_size=}")
        dataloaders.append(
            DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=batch_size,
                # drop_last=True,
            )
        )
    return dataloaders


def _prepare_dataloaders(
    img_dataset: Dataset[Tensor],
    distortions_group: DistortionsGroup,
    batch_size: int,
    sample_size: int,
    train_frac: float,
    validation_frac: float,
) -> tuple[
    list[DataLoader[Tensor]] | None,
    list[DataLoader[Tensor]] | None,
    list[DataLoader[Tensor]] | None,
]:
    datasets_train: list[Dataset[Tensor]] = []
    datasets_validation: list[Dataset[Tensor]] = []
    datasets_test: list[Dataset[Tensor]] = []

    for dataset in [img_dataset] + distortions_group.datasets:
        if not dataset:  # because `dg.datasets` can be empty
            continue
        dataset_train, dataset_validation, dataset_test = random_split(
            dataset, train_frac, validation_frac
        )
        datasets_train.append(dataset_train)
        datasets_validation.append(dataset_validation)
        datasets_test.append(dataset_test)

    sample_size_train, sample_size_val, sample_size_test = _split_numbers(
        sample_size, train_frac=train_frac, validation_frac=validation_frac
    )
    dataloaders = (
        _as_dataloaders(
            datasets_train,
            batch_size=batch_size,
            sample_size=sample_size_train,
        ),
        _as_dataloaders(
            datasets_validation,
            batch_size=batch_size,
            sample_size=sample_size_val,
        ),
        _as_dataloaders(
            datasets_test, batch_size=batch_size, sample_size=sample_size_test
        ),
    )
    return dataloaders


def train(
    model: nn.Module,
    img_dataset: Dataset[Tensor],
    random_seed: int,
    batch_size: int,
    sample_size: int,
    train_frac: float,
    validation_frac: float,
    epochs: int,
    epoch_dir: Path,
    create_optimizer: Callable[[nn.Module], torch.optim.Optimizer],
    loss_function: LossFunction,
    distortions_group: DistortionsGroup,
    no_progress: bool,
    patience: int,
) -> None:
    epoch_dir.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    dls_train, dls_validation, dls_test = _prepare_dataloaders(
        img_dataset=img_dataset,
        distortions_group=distortions_group,
        batch_size=batch_size,
        sample_size=sample_size,
        train_frac=train_frac,
        validation_frac=validation_frac,
    )
    ci.log(f"{sample_size=} " f"{train_frac=} " f"{validation_frac=}")

    def _log_size(name: str, dls: list[DataLoader[Any]] | None):
        if dls is None:
            ci.log(f"{name}: [red]disabled")
        else:
            ci.log(f"{name}: {[len(dl) for dl in dls]} batches")

    _log_size("Train", dls_train)
    _log_size("Validation", dls_validation)
    _log_size("Test", dls_test)

    optimizer = create_optimizer(model)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn("loss: {task.fields[loss]}"),
        disable=no_progress,
    ) as p:
        epoch_task = p.add_task("Epoch...", total=epochs, loss="?")

        if dls_train is not None:  # allow "test only" mode
            try:
                _train_loop(
                    p,
                    model=model,
                    epochs=epochs,
                    dls_train=dls_train,
                    dls_validation=dls_validation,
                    epoch_task=epoch_task,
                    optimizer=optimizer,
                    epoch_dir=epoch_dir,
                    loss_function=loss_function,
                    distortions_group=distortions_group,
                    patience=patience,
                )
            except KeyboardInterrupt:
                p.log("training stopped by user (Ctrl+C)")

        p.print(p)
        p.remove_task(epoch_task)

        # test
        if dls_test is not None:
            test_task = p.add_task(
                "Test... ", total=len(dls_test[0]), loss="?"
            )
            test_loss = 0.0
            with torch.inference_mode():
                for loss in p.track(
                    _evaluate_dataset(
                        model,
                        dls=dls_test,
                        distortions_group=distortions_group,
                        loss_function=loss_function,
                    ),
                    task_id=test_task,
                ):
                    test_loss += loss.item()
                    p.update(test_task, loss=f"{test_loss:g}")
            test_loss /= len(dls_test[0])
            p.update(test_task, loss=f"{test_loss:g}")
            p.console.print(p, end="")
            p.remove_task(test_task)


def dicts_merge(d: dict[str, Any], u: dict[str, Any]):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = dicts_merge(d[k], v)
        else:
            d[k] = u[k]
    return d


def parse_config() -> tuple[Config, bool]:
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
        help="create json schema file (schema.json)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="don't show progress bar, helpful when debugging with pdb",
    )
    parser.add_argument(
        "--override",
        type=json5.loads,
        help=(
            "Override values from --config, for example, to change "
            'the number of epochs, pass {"epochs": 200}'
        ),
        default={},
    )
    args = parser.parse_args()

    if args.update_schema:
        schema_path = Path(__file__).with_name("schema.json")
        schema_path.write_text(
            json5.dumps(Config.model_json_schema(), ensure_ascii=False)
        )
        ci.console.log(f"[green] {schema_path} [cyan]saved")
        raise SystemExit(0)
    ci.console.log(f"Using [green]{args.config}")

    with args.config.open() as f:
        data: dict[str, Any] = json5.load(f)
    data = dicts_merge(data, args.override)
    ci.console.print_json(data=data)
    config = Config(**data)
    return config, args.no_progress


def main():
    config, no_progress = parse_config()
    if torch.cuda.is_available():
        device_str = "cuda"
        ci.log("Current device: [bold green]GPU")
    else:
        device_str = "cpu"
        ci.log("Current device: [bold red]CPU")

    @contextmanager
    def download_progress():
        def progress_callback(description: str, done: float):
            progress.update(task1, completed=done, description=description)

        yield progress_callback

    with torch.device(device_str) as device:
        with Progress(disable=no_progress) as progress:

            task1 = progress.add_task("Dataset...", total=1.0)
            distortions_group = config.load_distortions(download_progress())
            model = config.model.get_instance()
            model = model.to(device)  # type: ignore
            loss_function = config.loss_function.load(model)

            img_dataset = config.img.load(download_progress())
            progress.update(task1, completed=1.0)
        create_optimizer = config.optimizer.load()
        train(
            model,
            img_dataset,
            random_seed=config.random_seed,
            batch_size=config.batch_size,
            train_frac=config.train_frac,
            sample_size=config.sample_size,
            validation_frac=config.validation_frac,
            epochs=config.epochs,
            epoch_dir=config.epoch_dir,
            create_optimizer=create_optimizer,
            loss_function=loss_function,
            distortions_group=distortions_group,
            no_progress=no_progress,
            patience=config.patience,
        )


if __name__ == "__main__":
    main()
