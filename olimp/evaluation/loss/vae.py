import torch
from torch import Tensor


def vae_loss(
    pred: Tensor, target: Tensor, mu: Tensor, logvar: Tensor
) -> Tensor:
    """
    Implementation of Variational autoencoder loss
    """
    L1 = torch.nn.functional.l1_loss(pred, target, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L1 + KLD
