from __future__ import annotations
from typing import cast, Iterator, Literal, Callable, NewType
from pathlib import Path
import numpy as np
import os
from torch import Tensor, tensor
from torch._prims_common import DeviceLikeType
from torchvision.io import read_image


SubPath = NewType("SubPath", str)


class ZenodoItem:
    def __init__(self, path: Path) -> None:
        self.path = path

    def data(self, device: DeviceLikeType = "cpu") -> Tensor:
        """
        Default device is "cpu" because it's the torch way
        """
        if self.path.suffix == ".jpg":
            return tensor(read_image(self.path), device=device)
        elif self.path.suffix == ".csv":
            return tensor(
                np.loadtxt(self.path, delimiter=",", dtype=np.float32),
                device=device,
            ).unsqueeze(0)
        else:
            raise ValueError(
                f"internal olimp error. Didn't expect {self.path}"
            )


def _download_zenodo(
    root: Path,
    record: Literal[7848576],
    progress_callback: Callable[[str, float], None] | None,
) -> None:
    import requests
    from zipfile import ZipFile

    r = requests.get(f"https://zenodo.org/api/records/{record}")
    for file in r.json()["files"]:
        name = cast(str, file["key"])  # "SCA-2023.zip"
        url = cast(str, file["links"]["self"])
        zip_path = root / name
        if not zip_path.exists():
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                downloaded = 0.0
                with zip_path.open("wb") as out_zip:
                    for chunk in r.iter_content(chunk_size=0x10000):
                        out_zip.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(
                                f"Downloading {name}",
                                downloaded
                                / float(r.headers["Content-Length"]),
                            )
        assert zip_path.exists(), zip_path

        print("zip_path", zip_path)
        with ZipFile(zip_path) as zf:
            for idx, member in enumerate(zf.infolist(), 1):
                zf.extract(member, root)
                if progress_callback:
                    progress_callback(
                        f"Unpacking {name}", idx / len(zf.infolist())
                    )


def _read_dataset_dir(
    dataset_root: Path, subpaths: set[SubPath]
) -> Iterator[tuple[SubPath, list[ZenodoItem]]]:
    from os import walk

    # this code can be simpler, going through all subpaths,

    for root, dirs, files in walk(dataset_root, onerror=print):
        root = Path(root)
        subpath = SubPath(
            str(root.relative_to(dataset_root)).replace("\\", "/")
        )
        fsubpaths = [sp for sp in subpaths if subpath.startswith(sp)]
        if "*" in subpaths:  # special case
            fsubpaths.append(SubPath("*"))
        if not fsubpaths:
            continue
        good_paths = [
            file for file in files if file.endswith((".jpg", ".jpeg"))
        ] or [
            file
            for file in files
            if file.endswith(".csv") and file != "parameters.csv"
        ]
        if good_paths:
            items = [ZenodoItem(root / file) for file in files]
            for subpath in fsubpaths:
                yield subpath, items


progress = None


def default_progress(action: str, done: float) -> None:
    """
    suitable for demo purposes only
    """
    global progress, task1
    if not progress:
        from rich.progress import Progress

        progress = Progress()
        progress.start()
        task1 = progress.add_task("Dataset...", total=1.0)

    progress.update(task1, completed=done, description=action)


def load_dataset(
    dataset_name: Literal["SCA-2023", "OLIMP"],
    record: Literal[7848576, 13692233],
    subpaths: set[SubPath],
    progress_callback: Callable[[str, float], None] | None = default_progress,
) -> dict[SubPath, list[ZenodoItem]]:
    root_path = Path(os.environ.get("OLIMP_DATATEST", ".datasets")).absolute()
    dataset_path = root_path / dataset_name
    if not dataset_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)
        _download_zenodo(
            root_path, record=record, progress_callback=progress_callback
        )

    dataset: dict[SubPath, list[ZenodoItem]] = {}
    for subpath, items in _read_dataset_dir(dataset_path, subpaths):
        if subpath in dataset:
            dataset[subpath] += items
        else:
            dataset[subpath] = items
    return dataset
