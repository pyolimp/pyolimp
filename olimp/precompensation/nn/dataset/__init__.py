from __future__ import annotations
from typing import Generic, TypeVar

from torch import Tensor
from torch.utils.data import Dataset
from olimp.dataset._zenodo import ImgPath
from olimp.dataset import read_img_path, ProgressContext
from itertools import islice

SubPath = TypeVar("SubPath", covariant=True)


class BaseZenodoDataset(Dataset[Tensor], Generic[SubPath]):
    def __init__(
        self,
        subsets: set[SubPath] | None,
        progress_context: ProgressContext,
        limit: int | None = None,
    ):
        if subsets is None:
            subsets = getattr(self, "subsets", None)
            if not subsets:
                raise ValueError("Specify subsets or use predefined classes")

        dataset = self.create_dataset(
            categories=subsets, progress_context=progress_context
        )
        self._items = list(
            islice(
                (item for subset in subsets for item in dataset[subset]), limit
            )
        )

    def create_dataset(
        self,
        categories: set[SubPath],
        progress_context: ProgressContext,
    ) -> dict[SubPath, list[ImgPath]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tensor:
        return read_img_path(self._items[index])

    def __len__(self):
        return len(self._items)
