from __future__ import annotations
from unittest import TestCase
from olimp.evaluation.loss import ms_ssim
import torch


class TestMSSSIM(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        # Calculate the MS-SSIM loss
        loss = ms_ssim(pred, target)
        print(loss)
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
