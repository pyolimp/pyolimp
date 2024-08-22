from __future__ import annotations
from unittest import TestCase
from olimp.evaluation.loss import ms_ssim, RMS, CD
import torch


class TestMSSSIM(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        # Calculate the MS-SSIM loss
        loss = ms_ssim(pred, target)

        # Assert that the loss is zero
        self.assertEqual(loss, 0)

    def test_empty_zero_and_ones_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.ones((1, 3, 256, 256))

        # Calculate the MS-SSIM loss
        loss = ms_ssim(pred, target)

        # Assert that the loss is non-zero
        self.assertNotEqual(loss, 0)


class TestRMS(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        # Calculate the RMS loss
        loss = RMS(pred, target, "lab")

        # Assert that the loss is zero
        self.assertEqual(loss, 0)

    def test_empty_zero_and_rand_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        seed = hash(torch.mean(pred).item())
        rng = torch.Generator().manual_seed(seed)
        target = torch.rand(1, 3, 256, 256, generator=rng)

        # Calculate the RMS loss
        loss = RMS(pred, target, "lab")

        # Assert that the loss is non-zero
        self.assertNotEqual(loss, 0)


class TestCD(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        # Calculate the CD loss
        loss = CD(pred, target, "lab")

        # Assert that the loss is zero
        self.assertEqual(loss, 0)

    def test_empty_zero_and_ones_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.ones((1, 3, 256, 256))

        # Calculate the CD loss
        loss = CD(pred, target, "lab")

        # Assert that the loss is non-zero
        self.assertNotEqual(loss, 0)
