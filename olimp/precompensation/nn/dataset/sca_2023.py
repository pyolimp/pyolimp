from __future__ import annotations
from typing import Callable
from . import BaseZenodoDataset, ImgPath
from olimp.dataset.sca_2023 import sca_2023 as _sca_2023, Paths


class SCA2023Dataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: Callable[[str, float], None] | None,
    ) -> dict[Paths, list[ImgPath]]:
        return _sca_2023(
            categories=categories, progress_callback=progress_callback
        )


class SCA2023DatasetImages(SCA2023Dataset):
    subsets: set[Paths] = {"Images"}


class SCA2023DatasetPSF(SCA2023Dataset):
    subsets: set[Paths] = {"PSFs"}
