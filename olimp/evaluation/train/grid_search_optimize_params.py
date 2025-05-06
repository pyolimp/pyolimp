from __future__ import annotations

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch._prims_common import DeviceLikeType
from torch import Tensor
from pathlib import Path
from itertools import product
from typing import Any, cast


# Dataset that computes the difference between metrics of two images
class CSVDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        csv_path: Path,
        numeric_cols: list[str],
        device: DeviceLikeType = "cpu",
    ):
        self.numeric_cols = [col.replace("-", "_") for col in numeric_cols]
        self.samples: list[dict[str, Tensor]] = []
        self.device = device

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    col.replace("-", "_"): torch.tensor(
                        float(row[f"{col}_2"]) - float(row[f"{col}_1"]),
                        device=self.device,
                    )
                    for col in numeric_cols
                }
                sample["target"] = torch.tensor(
                    float(row["score2_norm"]) - float(row["score1_norm"]),
                    device=self.device,
                )
                if sample["target"].item() != 0:
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.samples[idx]


# Parse the formula to extract coefficient and data variable names
def _parse_formula(formula: str) -> tuple[list[str], list[str], str]:
    import re

    coeff_vars = sorted(set(re.findall(r"\b[a-d]\b", formula)))
    data_vars = sorted(
        set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula)) - set(coeff_vars)
    )
    return coeff_vars, data_vars, formula


# Evaluate sign agreement accuracy for a given set of weights
def _evaluate_weights(
    dataset: Dataset[dict[str, Tensor]],
    coeff_vars: list[str],
    data_vars: list[str],
    formula: str,
    weights: tuple[float, ...],
    device: DeviceLikeType,
    batch_size: int = 1000,
):
    weights_tensor: Tensor = torch.tensor(weights, device=device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=lambda x: x
    )
    correct = 0
    total = 0

    formula_code = compile(formula, "<string>", "eval")

    for batch in dataloader:
        metrics: Tensor = torch.stack(
            [
                torch.stack([sample[var] for var in data_vars])
                for sample in batch
            ]
        ).to(device)
        targets = torch.stack([sample["target"] for sample in batch]).to(
            device
        )

        locals: dict[str, Any] = {"__builtins__": __builtins__}
        for i, var in enumerate(coeff_vars):
            locals[var] = weights_tensor[i]
        for i, var in enumerate(data_vars):
            locals[var] = metrics[:, i]
        metric_diff = eval(
            formula_code, {"__builtins__": __builtins__}, locals
        )

        correct += torch.count_nonzero(
            torch.sign(metric_diff) == torch.sign(targets)
        )
        total += targets.size(0)

    return correct / total if total > 0 else 0


# Brute-force weight optimization (coarse and fine steps)
def brute_force_optimize(
    dataset: Dataset[dict[str, Tensor]],
    formula: str,
    step_coarse: float = 0.01,
    step_fine: float = 0.001,
    device: DeviceLikeType = "cpu",
    batch_size: int = 1000,
) -> tuple[dict[str, Any], float]:
    coeff_vars, data_vars, parsed_formula = _parse_formula(formula)
    num_vars = len(coeff_vars)

    if num_vars == 0:
        raise ValueError(
            "The formula does not contain any coefficients to optimize."
        )
    if num_vars > 4:
        raise ValueError(
            f"The formula should not contain more than 4 coefficients, got: {num_vars}"
        )

    if len(dataset) == 0:
        raise ValueError(
            "The dataset is empty — no valid rows after filtering."
        )

    best_weights, best_acc = None, -1
    all_results: list[tuple[tuple[float, ...], float]] = []

    for weights in product(
        cast(
            list[float], torch.arange(0, 1 + step_coarse, step_coarse).tolist()
        ),
        repeat=num_vars,
    ):
        if sum(weights) > 1:
            continue
        acc = _evaluate_weights(
            dataset,
            coeff_vars,
            data_vars,
            parsed_formula,
            weights,
            device,
            batch_size,
        )
        all_results.append((weights, acc))
        print(f"Trial: {dict(zip(coeff_vars, weights))} → accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_weights = acc, weights

    if best_weights is None:
        raise ValueError(
            "No suitable weights found during coarse step. Check your data and formula."
        )

    fine_ranges = [
        cast(
            list[float],
            torch.arange(
                max(0, w - step_coarse),
                min(1, w + step_coarse) + 1e-6,
                step_fine,
            ).tolist(),
        )
        for w in best_weights
    ]

    for weights in product(*fine_ranges):
        if sum(weights) > 1:
            continue
        acc = _evaluate_weights(
            dataset,
            coeff_vars,
            data_vars,
            parsed_formula,
            weights,
            device,
            batch_size,
        )
        all_results.append((weights, acc))
        print(
            f"Refinement: {dict(zip(coeff_vars, weights))} → accuracy: {acc:.4f}"
        )
        if acc > best_acc:
            best_acc, best_weights = acc, weights

    best_weights_dict = dict(zip(coeff_vars, best_weights))
    print("\n==== Final Result ====")
    print(f"Best weights: {best_weights_dict}")
    print(f"Accuracy (sign agreement): {best_acc:.4f}")

    return best_weights_dict, best_acc


# Main script for execution
if __name__ == "__main__":
    csv_path = Path("directional_agreement_report.csv")

    # Formula with coefficients and metrics from CSV
    formula = "a * ms_ssim + (1 - a) * corr"
    numeric_cols = ["ms-ssim", "corr"]

    # Alternative example:
    # formula = "a * nrmse + b * stress + (1 - a - b) * corr"
    # numeric_cols = ["nrmse", "stress", "corr"]

    dataset = CSVDataset(csv_path, numeric_cols, device="cuda")
    best_weights, best_accuracy = brute_force_optimize(
        dataset, formula, device="cuda", batch_size=1000
    )
