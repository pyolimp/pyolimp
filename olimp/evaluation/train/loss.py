import torch


def agreement_loss(
    metric_diff: torch.Tensor, score_diff: torch.Tensor, scale: float = 1000.0
) -> torch.Tensor:
    """
    Soft agreement loss: encourages agreement in sign between metric_diff and score_diff.
    """
    agree_soft = torch.sigmoid(scale * metric_diff * score_diff)
    return (1.0 - agree_soft).mean()


def correlation_loss(
    metric_diff: torch.Tensor, score_diff: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Minimize negative Pearson correlation between predicted and true score differences.
    """
    vx = metric_diff - metric_diff.mean()
    vy = score_diff - score_diff.mean()

    corr = (vx * vy).sum() / (
        torch.sqrt((vx**2).sum() + eps) * torch.sqrt((vy**2).sum() + eps)
    )
    return 1 - corr
