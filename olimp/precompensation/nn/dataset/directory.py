from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from collections.abc import Sequence, Iterator
from olimp.dataset import read_img_path, ImgPath


def read_path(root: Path, matches: Sequence[str]) -> Iterator[ImgPath]:
    from glob import iglob
    from fnmatch import fnmatch

    for str_path in iglob(f"{root}/**", recursive=True):
        for match in matches:
            if fnmatch(str_path, match):
                path = Path(str_path)
                if path.is_file():
                    yield ImgPath(path)


class DirectoryDataset(Dataset[Tensor]):
    def __init__(self, root: Path, matches: Sequence[str]) -> None:
        self._paths = list(read_path(root, matches))
        if not self._paths:
            raise ValueError(f"There are no {", ".join(matches)} in {root}")

    def __getitem__(self, index: int) -> Tensor:
        return read_img_path(self._paths[index])

    def __len__(self):
        return len(self._paths)