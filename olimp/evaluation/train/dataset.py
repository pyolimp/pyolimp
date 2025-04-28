from __future__ import annotations

from olimp.evaluation.loss.stress import STRESS
from olimp.evaluation.loss.corr import Correlation
from olimp.evaluation.loss.s_oklab import SOkLab
from olimp.evaluation.loss.flip import LDRFLIPLoss
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss
from olimp.evaluation.loss.lpips import LPIPS
from olimp.evaluation.loss.nrmse import NormalizedRootMSE

from torch import Tensor, device
from torch.utils.data import Dataset
from torchvision.io import read_image
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypedDict
import csv


class CSVRow(TypedDict):
    target_path: Path
    image1_path: Path
    image2_path: Path
    score1: float
    score2: float
    score1_norm: float
    score2_norm: float


class MetricItem(CSVRow):
    target_image: Tensor
    image1: Tensor
    image2: Tensor
    metric_values: dict[str, float]


def _pixels_to_inches(
    img_px_w: int | None = None,
    diagonal_inch: float = 24.0,
    aspect_w: int = 16,
    aspect_h: int = 9,
    img_percent_of_screen: float = 0.26,
) -> float:
    """
    Converts the width of an object from pixels or percent to inches.

    :param obj_px_w: object width in pixels
    :param diagonal_inch: screen diagonal in inches
    :param aspect_w: screen aspect ratio width
    :param aspect_h: screen aspect ratio height
    :param screen_width_px: screen width in pixels
    :param obj_percent: object width as a percentage of screen width
    :return: object width in inches
    """
    ratio = aspect_w / aspect_h
    screen_width_inch = (
        (diagonal_inch**2) * (ratio**2) / (1 + ratio**2)
    ) ** 0.5
    return img_px_w / (screen_width_inch * img_percent_of_screen)


@dataclass
class MetricInfo:
    name: str
    create_metric: Callable[[Tensor], Callable[[Tensor, Tensor], Tensor]]


def create_metrics(device: str | device = "cpu") -> list[MetricInfo]:
    ldr_flip_loss = LDRFLIPLoss().to(device)
    multi_scale_ssim_loss = MultiScaleSSIMLoss(reduction="mean").to(device)
    lpips = LPIPS(net="alex").to(device)
    soklab_cache: tuple[None, None] | tuple[float, SOkLab] = (None, None)

    def _create_soklab_metric(target: Tensor):
        nonlocal soklab_cache
        img_dpi_w = _pixels_to_inches(img_px_w=target.shape[-1])
        if soklab_cache[0] == img_dpi_w:
            soklab = soklab_cache[1]
            assert soklab is not None
        else:
            soklab = SOkLab(dpi=img_dpi_w, distance_inch=19.685).to(
                device=device
            )
            soklab_cache = (img_dpi_w, soklab)

        def soklab_metric(x: Tensor, y: Tensor) -> Tensor:
            return 1.0 - soklab(x, y)

        return soklab_metric

    return [
        MetricInfo(
            name="stress",
            create_metric=lambda _target: (
                STRESS(invert=True, reduction="mean")
            ),
        ),
        MetricInfo(
            name="corr",
            create_metric=lambda _target: (Correlation(invert=False)),
        ),
        MetricInfo(
            name="flip",
            create_metric=lambda _target: (
                lambda x, y: 1.0 - ldr_flip_loss(x, y)
            ),
        ),
        MetricInfo(
            name="ms-ssim",
            create_metric=lambda _target: (
                lambda x, y: 1.0 - multi_scale_ssim_loss(x, y)
            ),
        ),
        MetricInfo(
            name="lpips",
            create_metric=lambda _target: (lambda x, y: 1.0 - lpips(x, y)),
        ),
        MetricInfo(
            name="soklab",
            create_metric=_create_soklab_metric,
        ),
        MetricInfo(
            name="nrmse",
            create_metric=lambda _target: NormalizedRootMSE(invert=True),
        ),
    ]


class MetricDataset(Dataset[CSVRow]):
    def __init__(
        self,
        answers_dataset_paths: list[Path],
        image_dataset_paths: list[Path],
        metrics: list[MetricInfo] | None = None,
    ) -> None:
        """
        :param answers_dataset_paths:List of ways to obtain CSV files with attachments.
        :param image_dataset_paths: List of paths to folders with images (one CSV corresponds to one folder).
        :param transform: Transformations for images.
        """
        super().__init__()

        assert len(answers_dataset_paths) == len(
            image_dataset_paths
        ), "Each CSV must correspond to a folder of images"

        self.data: list[CSVRow] = []
        self.metrics = metrics

        for answers_base_path, image_base_path in zip(
            answers_dataset_paths, image_dataset_paths
        ):
            with open(
                answers_base_path, newline="", encoding="utf-8", mode="r"
            ) as csvfile:
                reader = csv.reader(csvfile)
                _header = next(reader)  # Skip the header

                for row in reader:
                    (
                        path1,
                        path2,
                        score1_str,
                        score2_str,
                        score1_norm_str,
                        score2_norm_str,
                    ) = row

                    path1 = Path(path1)
                    path2 = Path(path2)

                    score1 = float(score1_str)
                    score2 = float(score2_str)
                    score1_norm = float(score1_norm_str)
                    score2_norm = float(score2_norm_str)

                    self.data.append(
                        {
                            "target_path": image_base_path
                            / path1.parent
                            / "target.png",
                            "image1_path": image_base_path / path1,
                            "image2_path": image_base_path / path2,
                            "score1": score1,
                            "score2": score2,
                            "score1_norm": score1_norm,
                            "score2_norm": score2_norm,
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> MetricItem:
        item = self.data[idx]

        img1 = (
            read_image(str(item["image1_path"])).float() / 255.0
        ).unsqueeze(0)
        img2 = (
            read_image(str(item["image2_path"])).float() / 255.0
        ).unsqueeze(0)
        target = (
            read_image(str(item["target_path"])).float() / 255.0
        ).unsqueeze(0)

        metric_results = {}
        result: MetricItem = {
            **item,
            "target_image": target,
            "image1": img1,
            "image2": img2,
            "metric_values": metric_results,
        }

        for metric_info in self.metrics or ():
            metric_fn = metric_info.create_metric(target)
            try:
                value1 = metric_fn(target, img1)
                value2 = metric_fn(target, img2)
                metric_results[f"{metric_info.name}_1"] = float(value1.item())
                metric_results[f"{metric_info.name}_2"] = float(value2.item())
            except Exception as e:
                print(f"{metric_info.name} metric evaluation error: {e}")
                raise

        return result


def main():
    from .human_studies_download import human_studies_download

    human_studies = human_studies_download()
    dataset = MetricDataset(
        human_studies.answers_paths,
        human_studies.image_paths,
        metrics=create_metrics(),
    )
    sample = dataset[77]

    print(
        sample["target_path"],
        sample["image1_path"],
        sample["image2_path"],
        "\n",
        "Size of target image:",
        sample["target_image"].shape,
        "\n",
        "Size of image1:",
        sample["image1"].shape,
        "Score1:",
        sample["score1"],
        sample["score1_norm"],
        "\n",
        "Size of image2:",
        sample["image2"].shape,
        "Score2:",
        sample["score2"],
        sample["score2_norm"],
    )
    print(sample["metric_values"])


if __name__ == "__main__":
    main()
