from olimp.evaluation.train.dataset import MetricDataset, create_metrics

from pathlib import Path
import csv
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


def evaluate_directional_agreement(
    dataset: MetricDataset,
    threshold: float = 1e-5,
    csv_output_path: Path = Path("directional_agreement_report.csv"),
) -> dict[str, float]:
    metric_sums: dict[str, int] = {}
    metric_counts: dict[str, int] = {}
    rows: list[dict[str, str | float | Path]] = []
    fieldnames: set[str] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task(
            "[green]Evaluating agreement...", total=len(dataset)
        )

        for sample in dataset:
            score_diff = sample["score2_norm"] - sample["score1_norm"]
            row: dict[str, str | float | Path] = {
                "image1_path": sample["image1_path"],
                "image2_path": sample["image2_path"],
                "score1": sample["score1"],
                "score2": sample["score2"],
                "score1_norm": sample["score1_norm"],
                "score2_norm": sample["score2_norm"],
            }

            for key, value in sample["metric_values"].items():
                row[key] = value
                fieldnames.add(key)

            if abs(score_diff) >= threshold:
                for key in sample["metric_values"]:
                    if key.endswith("_1"):
                        name = key[:-2]
                        m1 = sample["metric_values"].get(f"{name}_1")
                        m2 = sample["metric_values"].get(f"{name}_2")
                        if m1 is None or m2 is None:
                            continue
                        metric_diff = m2 - m1
                        agree = (score_diff * metric_diff) > 0
                        row[f"agree_{name}"] = int(agree)
                        fieldnames.add(f"agree_{name}")

                        metric_sums[name] = metric_sums.get(name, 0) + agree
                        metric_counts[name] = metric_counts.get(name, 0) + 1

            rows.append(row)
            progress.advance(task)

    base_fields = [
        "image1_path",
        "image2_path",
        "score1",
        "score2",
        "score1_norm",
        "score2_norm",
    ]
    metric_fields = sorted(k for k in fieldnames if not k.startswith("agree_"))
    agree_fields = sorted(k for k in fieldnames if k.startswith("agree_"))
    all_fields = base_fields + metric_fields + agree_fields

    with open(csv_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        name: metric_sums[name] / metric_counts[name] for name in metric_sums
    }


if __name__ == "__main__":

    answers_paths = [
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/answers_datasets/testcomparervimethods.csv"
        ),
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/answers_datasets/testcomparervimetrics.csv"
        ),
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/answers_datasets/testcomparecorrmssim.csv"
        ),
    ]

    image_paths = [
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/images_datasets/testcomparervimethods"
        ),
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/images_datasets/testcomparervimetrics"
        ),
        Path(
            "/home/devel/olimp/pyolimp/olimp/evaluation/train/images_datasets/testcomparecorrmssim"
        ),
    ]

    dataset = MetricDataset(
        answers_paths, image_paths, metrics=create_metrics()
    )

    results = evaluate_directional_agreement(dataset)
    print("Directed consensus on metrics:")
    for metric, acc in results.items():
        print(f"{metric}: {acc}")
