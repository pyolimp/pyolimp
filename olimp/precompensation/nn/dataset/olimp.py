from __future__ import annotations
from typing import Callable
from . import BaseZenodoDataset, ImgPath
from olimp.dataset.olimp import olimp as _olimp, Paths


class OlimpDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: Callable[[str, float], None] | None,
    ) -> dict[Paths, list[ImgPath]]:
        return _olimp(
            categories=categories, progress_callback=progress_callback
        )


class OlimpImages(OlimpDataset):
    subsets: set[Paths] = {"*"}
