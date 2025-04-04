from __future__ import annotations
from typing import cast, Iterator, Literal, NewType
from pathlib import Path
import os
from types import TracebackType
from . import ImgPath, ProgressCallback, ProgressContext


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
                        progress_callback(
                            f"Downloading {name}",
                            downloaded / float(r.headers["Content-Length"]),
                        )
        assert zip_path.exists(), zip_path

        with ZipFile(zip_path) as zf:
            for idx, member in enumerate(zf.infolist(), 1):
                zf.extract(member, root)
                progress_callback(
                    f"Unpacking {name}", idx / len(zf.infolist())
                )
        # Remove file, so we know the download was successful
        zip_path.unlink()


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


class DefaultProgress:
    """
    suitable for demo purposes only
    """

    def __enter__(self) -> "DefaultProgress":
        from rich.progress import Progress

        self._progress = Progress()
        self._task1 = self._progress.add_task("Dataset...", total=1.0)
        self._progress.__enter__()
        return self

    def __call__(self, action: str, done: float) -> None:
        self._progress.update(self._task1, completed=done, description=action)

    def __exit__(
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        return self._progress.__exit__(typ, value, traceback)


default_progress = DefaultProgress()


def load_dataset(
    dataset_name_and_record: (
        tuple[Literal["SCA-2023"], Literal[7848576]]
        | tuple[Literal["OLIMP"], Literal[13692233]]
        | tuple[Literal["CVD"], Literal[13881170]]
    ),
    subpaths: set[SubPath],
    progress_context: ProgressContext = default_progress,
) -> dict[SubPath, list[ImgPath]]:
    dataset_name, record = dataset_name_and_record

    cache_root = os.getenv("OLIMP_DATASETS")
    if cache_root is None:
        cache_home_dir = os.getenv("XDG_CACHE_HOME")
        if not cache_home_dir:
            cache_home_dir = Path("~/.cache").expanduser()
        cache_root = Path(cache_home_dir) / "pyolimp"
    dataset_path = Path(cache_root) / dataset_name

    with progress_context as progress:
        if not dataset_path.exists():
            print(f"downloading dataset to {dataset_path}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            _download_zenodo(
                dataset_path, record=record, progress_callback=progress
            )
        else:
            zips = list(dataset_path.glob("*.zip"))

            if zips:
                print("Dataset is corrupted")
                print(f"Redownloading dataset to {dataset_path}")
                # zips = corrupt download
                for zipfile in zips:
                    zipfile.unlink()
                _download_zenodo(
                    dataset_path,
                    record=record,
                    progress_callback=progress,
                )

        dataset: dict[SubPath, list[ImgPath]] = {}
        for subpath, items in _read_dataset_dir(dataset_path, subpaths):
            if subpath in dataset:
                dataset[subpath] += items
            else:
                dataset[subpath] = items
        progress(f"Loaded {len(dataset)} subsets", 1.0)
    return dataset
