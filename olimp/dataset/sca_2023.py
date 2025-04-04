from __future__ import annotations
from typing import Literal, TypeVar, cast
from ._zenodo import load_dataset, SubPath, default_progress
from . import read_img_path, ImgPath, ProgressContext

Paths = Literal[
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

T = TypeVar("T", bound=Paths)


def sca_2023(
    categories: set[T],
    progress_context: ProgressContext = default_progress,
) -> dict[T, list[ImgPath]]:
    """
    Downloads full dataset from https://zenodo.org/records/7848576

    Returns a dictionary of category -> list of paths
    """
    dataset = load_dataset(
        ("SCA-2023", 7848576),
        cast(set[SubPath], categories),
        progress_context=progress_context,
    )
    return cast(dict[T, list[ImgPath]], dataset)


if __name__ == "__main__":
    try:
        dataset = sca_2023(categories={"Images", "PSFs/Medium"})
    finally:
        from ._zenodo import progress

        if progress:
            progress.stop()
    print(sorted(dataset))
    print(read_img_path(dataset["Images"][0]).shape)
    print(read_img_path(dataset["PSFs/Medium"][0]).shape)
