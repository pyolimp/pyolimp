from typing import NamedTuple
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


class HumanStudies(NamedTuple):
    answers_paths: list[Path]
    image_paths: list[Path]


def _download_answers(
    csv_filename: str, local_files_only: bool = False
) -> Path:
    return Path(
        hf_hub_download(
            repo_id="pyolimp/human-studies",
            repo_type="dataset",
            filename=f"answers_datasets/{csv_filename}",
            local_files_only=local_files_only,
        )
    )


def _snapshot_download(name: str, local_files_only: bool = False) -> Path:
    return Path(
        snapshot_download(
            repo_id="pyolimp/human-studies",
            repo_type="dataset",
            allow_patterns=f"images_datasets/{name}/*",
            max_workers=2,  # against "429 Client Error: Too Many Requests"
            local_files_only=local_files_only,
        )
    )


def download_answers(csv_filename: str) -> Path:
    try:
        return _download_answers(csv_filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return _download_answers(csv_filename)


def download_dataset(name: str) -> Path:
    try:
        path = _snapshot_download(name, local_files_only=True)
    except LocalEntryNotFoundError:
        path = _snapshot_download(name)
    return path / "images_datasets" / name


def human_studies_download() -> HumanStudies:

    answers_paths = [
        download_answers("testcomparervimethods.csv"),
        download_answers("testcomparervimetrics.csv"),
        download_answers("testcomparecorrmssim.csv"),
    ]

    image_paths = [
        download_dataset("testcomparervimethods"),
        download_dataset("testcomparervimetrics"),
        download_dataset("testcomparecorrmssim"),
    ]
    return HumanStudies(answers_paths=answers_paths, image_paths=image_paths)
