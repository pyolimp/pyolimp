from __future__ import annotations
from typing import cast, Iterator, Literal, Callable
from functools import cached_property
from pathlib import Path
import numpy as np
import os


Paths = Literal[
    "abstracts and textures/abstract art",
    "abstracts and textures/backgrounds and patterns",
    "abstracts and textures/colorful abstracts",
    "abstracts and textures/geometric shapes",
    "abstracts and textures/neon abstracts",
    "abstracts and textures/textures",
    "animals/birds",
    "animals/farm animals",
    "animals/insects and spiders",
    "animals/marine life",
    "animals/pets",
    "animals/wild animals",
    "art and culture/cartoon and comics",
    "art and culture/crafts and handicrafts",
    "art and culture/dance and theater performances",
    "art and culture/music concerts and instruments",
    "art and culture/painting and frescoes",
    "art and culture/sculpture and bas-reliefs",
    "food and drinks/desserts and bakery",
    "food and drinks/dishes",
    "food and drinks/drinks",
    "food and drinks/food products on store shelves",
    "food and drinks/fruits and vegetables",
    "food and drinks/street food",
    "interiors/gyms and pools",
    "interiors/living spaces",
    "interiors/museums and galleries",
    "interiors/offices",
    "interiors/restaurants and cafes",
    "interiors/shopping centers and stores",
    "nature/beaches",
    "nature/deserts",
    "nature/fields and meadows",
    "nature/forest",
    "nature/mountains",
    "nature/water bodies",
    "objects and items/books and stationery",
    "objects and items/clothing and accessories",
    "objects and items/electronics and gadgets",
    "objects and items/furniture and decor",
    "objects and items/tools and equipment",
    "objects and items/toys and games",
    "portraits and people/athletes and dancers",
    "portraits and people/crowds and demonstrations",
    "portraits and people/group photos",
    "portraits and people/individual portraits",
    "portraits and people/models on runway",
    "portraits and people/workers in their workplaces",
    "sports and active leisure/cycling and rollerblading",
    "sports and active leisure/extreme sports",
    "sports and active leisure/individual sports",
    "sports and active leisure/martial arts",
    "sports and active leisure/team sports",
    "sports and active leisure/tourism and hikes",
    "text and pictogram/billboard text",
    "text and pictogram/blueprints",
    "text and pictogram/caricatures and pencil drawing",
    "text and pictogram/text documents",
    "text and pictogram/traffic signs",
    "urban scenes/architecture",
    "urban scenes/city at night",
    "urban scenes/graffiti and street art",
    "urban scenes/parks and squares",
    "urban scenes/streets and avenues",
    "urban scenes/transport",
]


class OLIMPItem:
    def __init__(self, path: Path) -> None:
        self.path = path

    @cached_property
    def data(self):
        if self.path.suffix in (".jpg", ".jpeg"):
            from PIL import Image

            return np.asarray(Image.open(self.path))
        elif self.path.suffix == ".csv":
            return np.loadtxt(self.path, delimiter=",")
        else:
            raise ValueError(
                f"internal olimp error. Didn't expect {self.path}"
            )


def _download_olimp(
    root: Path, progress_callback: Callable[[str, float], None] | None
) -> None:
    import requests
    from zipfile import ZipFile

    r = requests.get("https://zenodo.org/api/records/13692233")
    for file in r.json()["files"]:
        name = cast(str, file["key"])  # "olimp.zip"
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
                                "Downloading OLIMP",
                                downloaded
                                / float(r.headers["Content-Length"]),
                            )
        assert zip_path.exists(), zip_path

        with ZipFile(zip_path) as zf:
            for idx, member in enumerate(zf.infolist(), 1):
                zf.extract(member, root)
                if progress_callback:
                    progress_callback(
                        "Unpacking OLIMP", idx / len(zf.infolist())
                    )


def _read_dataset_dir(
    dataset_root: Path,
) -> Iterator[tuple[Paths, list[OLIMPItem]]]:
    from os import walk

    for root, dirs, files in walk(dataset_root, onerror=print):
        good_paths = [
            file for file in files if file.endswith((".jpg", "jpeg"))
        ]
        if not good_paths:
            good_paths = [
                file
                for file in files
                if file.endswith(".csv") and file != "parameters.csv"
            ]
        if good_paths:
            root = Path(root)
            items = [OLIMPItem(root / file) for file in files]
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
        task1 = progress.add_task("OLIMP...", total=1.0)

    progress.update(task1, completed=done, description=action)


def olimp(
    progress_callback: Callable[[str, float], None] | None = default_progress
) -> dict[Paths, list[OLIMPItem]]:
    root_path = Path(os.environ.get("OLIMP_DATATEST", ".datasets")).absolute()
    dataset_path = root_path / "OLIMP"
    if not dataset_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)
        _download_olimp(root_path, progress_callback)

    dataset = dict(_read_dataset_dir(dataset_path))

    return dataset


if __name__ == "__main__":
    try:
        dataset = olimp()
    finally:
        if progress:
            progress.stop()
    print(sorted(dataset))
    print(dataset["nature/beaches"][0].data.shape)
