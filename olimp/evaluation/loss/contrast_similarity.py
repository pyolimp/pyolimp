from __future__ import annotations

from ._base import ReducibleLoss
import torch
from torch import Tensor


class ContrastSimLoss(ReducibleLoss):
    """
    Computes Contrast Similarity metric between two images.
    """

    @staticmethod
    def _calc_gradx(image: Tensor) -> Tensor:
        grad_x = torch.empty_like(image)
        grad_x[:, :, :-1] = torch.subtract(image[:, :, 1:], image[:, :, :-1])
        grad_x[:, :, -1:] = torch.subtract(image[:, :, :1], image[:, :, -1:])
        return grad_x

    @staticmethod
    def _calc_grady(image: Tensor) -> Tensor:
        grad_y = torch.empty_like(image)
        grad_y[:, :-1, :] = torch.subtract(image[:, 1:, :], image[:, :-1, :])
        grad_y[:, -1:, :] = torch.subtract(image[:, :1, :], image[:, -1:, :])
        return grad_y

    @staticmethod
    def _calc_director_field(grad_x: Tensor, grad_y: Tensor) -> Tensor:
        # calculate symmetrical matrix
        # [[a, b]
        #  [b, c]]
        a = (grad_x * grad_x).sum(dim=0)
        b = (grad_y * grad_y).sum(dim=0)
        c = (grad_x * grad_y).sum(dim=0)
        # sympy.Matrix([[a,b], [b,c]]).eigenvals()
        # D1 - second (largest) Eighen value
        D1 = (
            a
            + b
            + torch.sqrt(
                torch.maximum(
                    # Determinant calculation
                    a**2 - 2 * a * b + b**2 + 4 * c**2,
                    # # Ensure non-negative values under the square root
                    torch.tensor(0.0),
                )
            )
        ) * 0.5
        c_not_zero = c != 0

        # Initialize the eigenvector array
        V1 = torch.zeros_like(D1)

        # Calculate the first eigenvector
        V1[c_not_zero] = (-b + D1)[c_not_zero] / c[c_not_zero]
        V1_len = torch.hypot(V1, torch.tensor(1))

        # Normalize the eigenvector and append a constant component
        largest_eigenvec2 = torch.dstack((V1 / V1_len, 1.0 / V1_len))

        # handle cases where c is zero
        largest_eigenvec2[~c_not_zero] = torch.dstack(
            ((a != 0), (a == 0))
        ).type(torch.float32)[~c_not_zero]

        largest_eigenvec2 *= torch.sqrt(D1)[..., None]

        return largest_eigenvec2

    @classmethod
    def _sign_solution_intensity(
        cls, image: Tensor, director_field: Tensor
    ) -> Tensor:  # Changing incorrect signs of the vector field
        mean_image = torch.sum(image, 0)
        vector_field = torch.zeros_like(director_field)
        J = torch.reshape(mean_image, (1,) + image.shape[1:])  # Mean image
        J_gradx = cls._calc_gradx(J)
        J_grady = cls._calc_grady(J)
        dot_x = torch.zeros(J_gradx.shape, dtype=torch.float32)
        dot_y = torch.zeros(J_grady.shape, dtype=torch.float32)

        director_fieldx = director_field[:, :, 0]
        director_fieldy = director_field[:, :, 1]

        dot_x = J_gradx * director_fieldx
        dot_y = J_grady * director_fieldy

        mean_image = dot_x + dot_y

        sign = torch.ones(mean_image.shape)
        sign[torch.where(mean_image < 0)] = -1

        vector_field[:, :, 0] = sign * director_field[:, :, 0]
        vector_field[:, :, 1] = sign * director_field[:, :, 1]

        return vector_field

    @classmethod
    def _vector_field_from_image(cls, image: Tensor) -> Tensor:
        grad_x = cls._calc_gradx(image)
        grad_y = cls._calc_grady(image)
        director_field = cls._calc_director_field(grad_x, grad_y)
        return cls._sign_solution_intensity(image, director_field)

    def _loss(self, x: Tensor, y: Tensor) -> Tensor:
        vf_x = self._vector_field_from_image(x)
        vf_y = self._vector_field_from_image(y)
        return torch.sum(
            torch.nan_to_num(
                torch.linalg.norm(vf_x - vf_y, ord=2, dim=2)
                / torch.linalg.norm(vf_y, ord=2, dim=2)
            )
        )
