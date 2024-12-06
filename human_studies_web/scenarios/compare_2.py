from __future__ import annotations
from pathlib import Path
from itertools import combinations
from random import Random


class Test1:
    """
    Информация о тесте.
    """

    def __init__(self) -> None:
        all_paths = (Path(__file__).parent / "test_images").glob("*.*")
        all_image_paths = [
            path
            for path in all_paths
            if path.is_file()
            and path.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        all_image_paths.sort()
        self._items = list(combinations(all_image_paths, 2))

    @staticmethod
    def render_test_for_user():
        return "Ready"

    def item_for_user(self, username: str, idx: int):
        files = Random(username).sample(self._items, k=5)[idx]
        return {
            "files": files,
            "choices": [
                "left",
                "right,
                "all good",
                "all bad"
                "unknown",
            ]
        }
