from __future__ import annotations
from typing import Callable
from unittest import TestCase
import torch
from torch import Tensor
from torch.testing import assert_close

Shape = tuple[int, int, int, int]


def _check_zero_zero(
    loss: Callable[[Tensor, Tensor], Tensor], shape: Shape = (2, 3, 256, 192)
) -> Tensor:
    pred = torch.zeros(shape, requires_grad=True)
    target = torch.zeros(shape, requires_grad=True)
    return loss(pred, target)


def _check_nonzero_nonzero(
    loss: Callable[[Tensor, Tensor], Tensor], shape: Shape = (2, 3, 128, 256)
) -> Tensor:
    pred = torch.zeros(shape)
    pred[0, 0, 16:48, 0:32] = 0.5
    pred.requires_grad_()
    target = torch.ones(shape)
    target[0, 0, 0:32, 0:32] = 0.5
    target.requires_grad_()
    return loss(pred, target)


class TestMSSSIM(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.piq import MultiScaleSSIMLoss

        loss = _check_zero_zero(MultiScaleSSIMLoss(reduction="none"))
        self.assertEqual(loss.tolist(), [0.0, 0.0])

    def test_empty_zero_and_ones_images(self):
        from olimp.evaluation.loss.piq import MultiScaleSSIMLoss

        loss = _check_nonzero_nonzero(
            MultiScaleSSIMLoss(reduction="none"), shape=(2, 3, 256, 256)
        )
        assert_close(loss, torch.tensor([0.707713723182, 0.706974804401]))
        self.assertTrue(loss.requires_grad)


class TestRMS(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.rms import RMS

        for color_space in ["lab", "prolab"]:
            loss = _check_zero_zero(RMS(color_space, reduction="none"))
            self.assertEqual(loss.tolist(), [0.0, 0.0])

    def test_lab_empty_zero_and_rand_images(self):
        from olimp.evaluation.loss.rms import RMS

        loss = _check_nonzero_nonzero(RMS("lab", reduction="none"))
        assert_close(loss, torch.tensor([0.06708120554685593, 0.0]))

    def test_prolab_empty_zero_and_rand_images(self):
        from olimp.evaluation.loss.rms import RMS

        loss = _check_nonzero_nonzero(RMS("prolab", reduction="none"))
        assert_close(loss, torch.tensor([0.0355894602835, 0.0]))
        self.assertTrue(loss.requires_grad)


class TestChromaticityDifference(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.chromaticity_difference import (
            ChromaticityDifference,
        )

        for color_space in ["lab", "prolab"]:
            assert color_space in ["lab", "prolab"]
            loss = _check_zero_zero(
                ChromaticityDifference(color_space, reduction="none")
            )
            self.assertEqual(loss.tolist(), [0.0, 0.0])

    def test_lab_empty_zero_and_ones_images(self):
        from olimp.evaluation.loss.chromaticity_difference import (
            ChromaticityDifference,
        )

        loss = _check_nonzero_nonzero(
            ChromaticityDifference("lab", reduction="none")
        )
        assert_close(loss, torch.tensor([0.0303995292633, 0.0]))

    def test_prolab_empty_zero_and_ones_images(self):
        from olimp.evaluation.loss.chromaticity_difference import (
            ChromaticityDifference,
        )

        loss = _check_nonzero_nonzero(
            ChromaticityDifference("prolab", reduction="none")
        )
        assert_close(loss, torch.tensor([0.040271583944, 3.65355923293e-8]))
        self.assertTrue(loss.requires_grad)


class TestPSNR(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.psnr import PSNR

        loss = _check_zero_zero(PSNR(reduction="none"))
        self.assertEqual(loss.tolist(), [float("inf")] * 2)

    def test_nonzero_nonzero(self):
        from olimp.evaluation.loss.psnr import PSNR

        loss = _check_nonzero_nonzero(PSNR(reduction="none"))
        assert_close(loss, torch.tensor([-5.96368026733, float("-inf")]))
        self.assertTrue(loss.requires_grad)


class TestSTRESS(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.stress import STRESS

        loss = _check_zero_zero(STRESS(invert=True, reduction="none"))
        self.assertEqual(loss.tolist(), [1.0, 1.0])

    def test_nonzero_nonzero(self):
        from olimp.evaluation.loss.stress import STRESS

        loss = _check_nonzero_nonzero(STRESS(invert=True, reduction="none"))
        assert_close(loss, torch.tensor([0.0029571056365966797, 1.0]))

        loss = _check_nonzero_nonzero(STRESS(invert=False, reduction="none"))
        assert_close(loss, torch.tensor([1.0 - 0.0029571056365966797, 0.0]))
        self.assertTrue(loss.requires_grad)


class TestCORR(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.corr import Correlation

        loss = _check_zero_zero(Correlation(invert=True, reduction="none"))
        self.assertEqual(loss.tolist(), [1.0, 1.0])

    def test_nonzero_nonzero(self):
        from olimp.evaluation.loss.corr import Correlation

        loss = _check_nonzero_nonzero(
            Correlation(invert=True, reduction="none")
        )
        assert_close(loss, torch.tensor([1.49472975730896, 1.0]))

        loss = _check_nonzero_nonzero(
            Correlation(invert=False, reduction="none")
        )
        self.assertTrue(loss.requires_grad)
        assert_close(loss, torch.tensor([-0.49472978711128235, 0.0]))


class TestSOkLab(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.s_oklab import SOkLab

        loss = _check_zero_zero(SOkLab(20.0, 39.0))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.tolist(), [0.0, 0.0])

    def test_nonzero_nonzero(self):
        from olimp.evaluation.loss.s_oklab import SOkLab

        loss = _check_nonzero_nonzero(SOkLab(20.0, 39.0))

        self.assertTrue(loss.requires_grad)
        assert_close(loss, torch.tensor([0.983500361442, 0.999999940395]))


class TestMSE(TestCase):
    def test_empty_zero_images(self):
        from olimp.evaluation.loss.mse import MSE

        loss = _check_zero_zero(MSE())
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss, 0.0)

    def test_nonzero_nonzero(self):
        from olimp.evaluation.loss.mse import MSE

        loss = _check_nonzero_nonzero(MSE())

        self.assertTrue(loss.requires_grad)
        assert_close(loss, torch.tensor(0.993489563465))
