from __future__ import annotations
from typing import Literal, TypeVar, cast
from ._zenodo import load_dataset, SubPath, default_progress
from . import read_img_path, ImgPath, ProgressContext

Paths = Literal[
    "Color_cvd_D_experiment_100000", "Color_cvd_P_experiment_100000", "*"
]

T = TypeVar("T", bound=Paths)


def cvd(
    categories: set[T],
    progress_context: ProgressContext = default_progress,
) -> dict[T, list[ImgPath]]:
    """
    Downloads full dataset from https://zenodo.org/records/13881170

    Returns a dictionary of category -> list of paths
    """
    dataset = load_dataset(
        ("CVD", 13881170),
        cast(set[SubPath], categories),
        progress_context=progress_context,
    )
    return cast(dict[T, list[ImgPath]], dataset)


if __name__ == "__main__":
    try:
        dataset = cvd(
            categories={
                "Color_cvd_D_experiment_100000",
                "Color_cvd_P_experiment_100000",
                "*",
            }
        )
    finally:
        from ._zenodo import progress

        if progress:
            progress.stop()
    print(sorted(dataset))
    print(read_img_path(dataset["Color_cvd_D_experiment_100000"][0]).shape)
    print(read_img_path(dataset["Color_cvd_P_experiment_100000"][0]).shape)
