from unittest import TestCase
from math import radians
import torch


def _mirror(a: torch.Tensor) -> torch.Tensor:
    ab = torch.hstack((a, torch.fliplr(a)))
    return torch.vstack((ab, torch.flipud(ab)))


class TestPSF(TestCase):
    def test_mirror(self):
        self.assertEqual(
            _mirror(torch.tensor([[1, 2], [3, 4]])).tolist(),
            [
                [1, 2, 2, 1],
                [3, 4, 4, 3],
                [3, 4, 4, 3],
                [1, 2, 2, 1],
            ],
        )

    def test_gauss(self):
        from olimp.simulate.psf_gauss import PSFGauss

        psf_gauss = PSFGauss(width=8, height=8)(
            center_x=4.0,
            center_y=4.0,
            theta=radians(45),
            sigma_x=1.0,
            sigma_y=1.0,
        )

        torch.testing.assert_close(
            psf_gauss,
            _mirror(
                torch.tensor(
                    [
                        [0.00, 0.10, 0.72, 1.97],
                        [0.10, 1.97, 14.53, 39.50],
                        [0.72, 14.53, 107.37, 291.85],
                        [1.97, 39.50, 291.85, 793.33],
                    ]
                )
                / 6400
            ),
        )

    def test_sca(self):
        from olimp.simulate.psf_sca import PSFSCA

        psf_sca = PSFSCA(width=8, height=8)(pupil_diameter_mm=1.5)

        torch.testing.assert_close(
            psf_sca,
            _mirror(
                torch.tensor(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 1],
                        [0, 1, 1, 1],
                    ],
                    dtype=torch.float32,
                )
                * 0.0416666679
            ),
        )
