from __future__ import annotations
from typing import cast, Iterator, Literal, Callable
from functools import cached_property
from pathlib import Path
import numpy as np
import os


Paths = Literal[
    "Images/Icons",
    "Images/Real_images/Animals",
    "Images/Real_images/Faces",
    "Images/Real_images/Natural",
    "Images/Real_images/Urban",
    "Images/Texts",
    "PSFs/Broad",
    "PSFs/Medium",
    "PSFs/Narrow",
]


class SCA2023Item:
    def __init__(self, path: Path) -> None:
        self.path = path

    @cached_property
    def data(self):
        if self.path.suffix == ".jpg":
            from PIL import Image

            return np.asarray(Image.open(self.path))
        elif self.path.suffix == ".csv":
            return np.loadtxt(self.path, delimiter=",")
        else:
            raise ValueError(
                f"internal olimp error. Didn't expect {self.path}"
            )


def _download_sca_2023(
    root: Path, progress_callback: Callable[[str, float], None] | None
) -> None:
    import requests
    from zipfile import ZipFile

    r = requests.get("https://zenodo.org/api/records/7848576")
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
                                "Downloading SCA-2023",
                                downloaded
                                / float(r.headers["Content-Length"]),
                            )
        assert zip_path.exists(), zip_path

        with ZipFile(zip_path) as zf:
            for idx, member in enumerate(zf.infolist(), 1):
                zf.extract(member, root)
                if progress_callback:
                    progress_callback(
                        "Unpacking SCA-2023", idx / len(zf.infolist())
                    )


def _read_dataset_dir(
    dataset_root: Path,
) -> Iterator[tuple[Paths, list[SCA2023Item]]]:
    from os import walk

    for root, dirs, files in walk(dataset_root, onerror=print):
        good_paths = [file for file in files if file.endswith(".jpg")]
        if not good_paths:
            good_paths = [
                file
                for file in files
                if file.endswith(".csv") and file != "parameters.csv"
            ]
        if good_paths:
            root = Path(root)
            items = [SCA2023Item(root / file) for file in files]
            path = cast(
                Paths, str(root.relative_to(dataset_root)).replace("\\", "/")
            )
            yield path, items


progress = None


def default_progress(action: str, done: float) -> None:
    global progress, task1
    if not progress:
        from rich.progress import Progress

        progress = Progress()
        progress.start()
        task1 = progress.add_task("SCA-2023...", total=1.0)

    progress.update(task1, completed=done, description=action)


def sca_2023(
    progress_callback: Callable[[str, float], None] | None = default_progress
) -> dict[Paths, list[SCA2023Item]]:
    root_path = Path(os.environ.get("OLIMP_DATATEST", ".datasets")).absolute()
    dataset_path = root_path / "SCA-2023"
    if not dataset_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)
        _download_sca_2023(root_path, progress_callback)

    dataset = dict(_read_dataset_dir(dataset_path))
    return dataset


if __name__ == "__main__":
    try:
        dataset = sca_2023()
    finally:
        if progress:
            progress.stop()
    print(sorted(dataset))
    print(dataset["Images/Icons"][0].data.shape)
    print(dataset["PSFs/Medium"][0].data.shape)
