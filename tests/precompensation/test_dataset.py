from olimp.precompensation.nn.dataset.directory import DirectoryDataset
from unittest import TestCase
from pathlib import Path


class TestDirectory(TestCase):
    path = Path(__file__).parents[1] / "test_data"

    def test_load(self):
        dataset = DirectoryDataset(self.path, ["*.jpg"])
        self.assertEqual(len(dataset), 2)

    def test_limit(self):
        dataset = DirectoryDataset(self.path, ["*.jpg"], limit=1)
        self.assertEqual(len(dataset), 1)

    def test_no_matches(self):
        with self.assertRaises(ValueError) as context:
            DirectoryDataset(self.path, ["*.zip"])
        self.assertEqual(
            str(context.exception), f"There are no *.zip in {self.path}"
        )

    def test_empty_matches(self):
        with self.assertRaises(ValueError) as context:
            DirectoryDataset(self.path, [])
        self.assertEqual(str(context.exception), "Matches must not be empty")
