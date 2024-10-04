from __future__ import annotations
from typing import Callable
from . import BaseZenodoDataset, ImgPath
from olimp.dataset.cvd import cvd as _cvd, Paths


class CDVDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: Callable[[str, float], None] | None,
    ) -> dict[Paths, list[ImgPath]]:
        return _cvd(categories=categories, progress_callback=progress_callback)
