from __future__ import annotations
from typing import Literal, TypeVar, cast, Callable
from ._zenodo import ZenodoItem, load_dataset, SubPath, default_progress

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
    progress_callback: Callable[[str, float], None] | None = default_progress,
) -> dict[T, list[ZenodoItem]]:
    dataset = load_dataset(
        "SCA-2023",
        7848576,
        cast(set[SubPath], categories),
        progress_callback=progress_callback,
    )
    return cast(dict[T, list[ZenodoItem]], dataset)


if __name__ == "__main__":
    try:
        dataset = sca_2023(categories={"Images", "PSFs/Medium"})
    finally:
        from ._zenodo import progress

        if progress:
            progress.stop()
    print(sorted(dataset))
    print(dataset["Images"][0].data.shape)
    print(dataset["PSFs/Medium"][0].data.shape)
