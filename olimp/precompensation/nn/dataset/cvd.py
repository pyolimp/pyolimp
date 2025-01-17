from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressCallback
from olimp.dataset.cvd import cvd as _cvd, Paths


class CVDDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: ProgressCallback,
    ) -> dict[Paths, list[ImgPath]]:
        return _cvd(categories=categories, progress_callback=progress_callback)
