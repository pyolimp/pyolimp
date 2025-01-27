from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressCallback
from olimp.dataset.sca_2023 import sca_2023 as _sca_2023, Paths


class SCA2023Dataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: ProgressCallback,
    ) -> dict[Paths, list[ImgPath]]:
        return _sca_2023(
            categories=categories, progress_callback=progress_callback
        )
