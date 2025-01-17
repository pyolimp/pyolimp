from __future__ import annotations
from typing import cast, Iterator, Literal, Callable, NewType
from pathlib import Path
import os
from . import ImgPath, ProgressCallback


SubPath = NewType("SubPath", str)


def _download_zenodo(
    root: Path,
    record: Literal[7848576, 13692233, 13881170],
    progress_callback: ProgressCallback,
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

        with ZipFile(zip_path) as zf:
            for idx, member in enumerate(zf.infolist(), 1):
                zf.extract(member, root)
                if progress_callback:
                    progress_callback(
                        f"Unpacking {name}", idx / len(zf.infolist())
                    )


def _read_dataset_dir(
    dataset_root: Path, subpaths: set[SubPath]
) -> Iterator[tuple[SubPath, list[ImgPath]]]:
    from os import walk

    # CVD dataset has two root files
    root_directories = [d for d in Path(dataset_root).iterdir() if d.is_dir()]
    if len(root_directories) == 1:
        (dataset_root,) = root_directories
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
            file for file in files if file.endswith((".jpg", ".jpeg", ".png"))
        ] or [
            file
            for file in files
            if file.endswith(".csv") and file != "parameters.csv"
        ]
        if good_paths:
            items = [ImgPath(root / file) for file in good_paths]
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
    dataset_name_and_record: (
        tuple[Literal["SCA-2023"], Literal[7848576]]
        | tuple[Literal["OLIMP"], Literal[13692233]]
        | tuple[Literal["CVD"], Literal[13881170]]
    ),
    subpaths: set[SubPath],
    progress_callback: ProgressCallback = default_progress,
) -> dict[SubPath, list[ImgPath]]:
    dataset_name, record = dataset_name_and_record
    root_path = Path(os.environ.get("OLIMP_DATATEST", ".datasets")).absolute()
    dataset_path = root_path / dataset_name
    if not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        _download_zenodo(
            dataset_path, record=record, progress_callback=progress_callback
        )

    dataset: dict[SubPath, list[ImgPath]] = {}
    for subpath, items in _read_dataset_dir(dataset_path, subpaths):
        if subpath in dataset:
            dataset[subpath] += items
        else:
            dataset[subpath] = items
    return dataset
