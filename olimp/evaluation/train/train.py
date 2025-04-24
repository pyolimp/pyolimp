import csv
import torch
from torch import nn, Tensor
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch._prims_common import DeviceLikeType

from olimp.evaluation.train.models.learned_formula import (
    LearnedMetricFromFormula,
)
from olimp.evaluation.train.loss import agreement_loss


class MetricPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        csv_path: Path,
        numeric_cols: list[str],
        device: DeviceLikeType = "cpu",
    ):
        self.numeric_cols = [col.replace("-", "_") for col in numeric_cols]
        self.samples: list[dict[str, torch.Tensor]] = []
        self.device = device

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sample = {
                        col.replace("-", "_"): torch.tensor(
                            float(row[f"{col}_2"]) - float(row[f"{col}_1"]),
                            device=self.device,
                        )
                        for col in numeric_cols
                    }

                    target = float(row["score2_norm"]) - float(
                        row["score1_norm"]
                    )
                    if abs(target) < 1e-5:
                        continue

                    sample["target"] = torch.tensor(target, device=self.device)
                    self.samples.append(sample)
                except Exception:
                    continue

        print(f"[MetricPairDataset] Loaded rows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


if __name__ == "__main__":

    csv_path = Path(
        "/home/devel/olimp/pyolimp/directional_agreement_report.csv"
    )
    numeric_cols = ["ms-ssim", "corr"]
    dataset = MetricPairDataset(csv_path, numeric_cols, device="cpu")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LearnedMetricFromFormula(
        formula="a * ms_ssim + (1 - b) * corr",
        learned_vars=["a", "b"],
        predictor_input_vars=["ms_ssim", "corr"],
        apply_sigmoid=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 1000

    # Create save directory and loss log
    save_dir = Path(__file__).parent / "weights"
    save_dir.mkdir(exist_ok=True)

    loss_log_path = save_dir / "loss_log.txt"
    best_agree = 0.0

    with open(loss_log_path, "w") as loss_log_file:
        for epoch in range(num_epochs):
            total_loss = 0.0
            agree_count = 0
            total = 0

            for batch in dataloader:
                input_vars = {
                    "ms_ssim": batch["ms_ssim"],
                    "corr": batch["corr"],
                }

                score_diff = batch["target"]
                metric_diff = model(input_vars)
                # print(metric_diff, score_diff)
                loss = agreement_loss(metric_diff, score_diff)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                agree = (metric_diff * score_diff) > 0
                agree_count += agree.sum().item()
                total += agree.numel()

            acc = agree_count / total if total > 0 else 0.0
            epoch_loss = total_loss / len(dataloader)

            print(
                f"Epoch {epoch+1}: Loss = {epoch_loss:.5f}, Agree = {acc:.5f}"
            )
            with open(loss_log_path, "a") as loss_log_file:
                loss_log_file.write(
                    f"{epoch+1}\t{epoch_loss:.6f}\t{acc:.6f}\n"
                )

            torch.save(
                model.state_dict(), save_dir / f"epoch_{epoch+1:04d}.pt"
            )

            if acc > best_agree:
                best_agree = acc
                torch.save(model.state_dict(), save_dir / "best.pt")
