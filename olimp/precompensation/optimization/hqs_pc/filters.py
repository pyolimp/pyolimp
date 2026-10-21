"""
Image Gradient Filters
"""

from __future__ import annotations
import torch
from torch import Tensor

SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

SOBEL_Y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

GRAD_X = [[-1, 1], [0, 0]]

GRAD_Y = [[1, 0], [-1, 0]]


def sobel(shape: torch.Size) -> Tensor:
    """
    Get Sobel filters
    """
    b, _, h, w = shape
    sobel = torch.zeros((b, 2, h, w))

    sobel[:, 0, :3, :3] = torch.tensor(SOBEL_X).expand(b, 3, 3)
    sobel[:, 1, :3, :3] = torch.tensor(SOBEL_Y).expand(b, 3, 3)

    return sobel


def grad(shape: torch.Size) -> Tensor:
    """
    Get gradient filters
    """
    b, _, h, w = shape
    grad = torch.zeros((b, 2, h, w))

    grad[:, 0, :2, :2] = torch.tensor(GRAD_X).expand(b, 2, 2)
    grad[:, 1, :2, :2] = torch.tensor(GRAD_Y).expand(b, 2, 2)

    return grad


def grad_like(input: Tensor) -> Tensor:
    """
    Get gradient filters with `input` shape & device.
    """
    b, _, h, w = input.shape
    grad = torch.zeros((b, 2, h, w))

    grad[:, 0, :2, :2] = torch.tensor(GRAD_X).expand(b, 2, 2)
    grad[:, 1, :2, :2] = torch.tensor(GRAD_Y).expand(b, 2, 2)

    grad = grad.to(input.device)

    return grad


def sobel_like(input: Tensor) -> Tensor:
    """
    Get Sobel filters with `input` shape & device.
    """
    b, _, h, w = input.shape
    sobel = torch.zeros((b, 2, h, w))

    sobel[:, 0, :3, :3] = torch.tensor(SOBEL_X).expand(b, 3, 3)
    sobel[:, 1, :3, :3] = torch.tensor(SOBEL_Y).expand(b, 3, 3)

    sobel = sobel.to(input.device)

    return sobel
