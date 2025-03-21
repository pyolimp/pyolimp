from __future__ import annotations
from unittest import TestCase
from olimp.evaluation.loss.chromaticity_difference import (
    ChromaticityDifference,
)
from olimp.evaluation.loss.rms import RMS
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss
import torch


class TestMSSSIM(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        # Calculate the MS-SSIM loss
        loss = MultiScaleSSIMLoss()(pred, target)

        # Assert that the loss is zero
        self.assertEqual(loss, 0)

    def test_empty_zero_and_ones_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.ones((1, 3, 256, 256))

        # Calculate the MS-SSIM loss
        loss = MultiScaleSSIMLoss()(pred, target)

        # Assert that the loss is non-zero
        self.assertNotEqual(loss, 0)


class TestRMS(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        for color_space in ["lab", "prolab"]:
            # Calculate the RMS loss
            loss = RMS(color_space)(pred, target)

            # Assert that the loss is zero
            self.assertEqual(loss, 0)

    def test_empty_zero_and_rand_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        seed = hash(torch.mean(pred).item())
        rng = torch.Generator().manual_seed(seed)
        target = torch.rand(1, 3, 256, 256, generator=rng)

        for color_space in ["lab", "prolab"]:
            # Calculate the RMS loss
            loss = RMS(color_space)(pred, target)

            # Assert that the loss is non-zero
            self.assertNotEqual(loss, 0)


class TestChromaticityDifference(TestCase):
    def test_empty_zero_images(self):
        # Create two empty zero images
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.zeros((1, 3, 256, 256))

        for color_space in ["lab", "prolab"]:
            # Calculate the CDLab loss
            loss = ChromaticityDifference(color_space)(pred, target)

            # Assert that the loss is zero
            self.assertEqual(loss, 0)

    def test_empty_zero_and_ones_images(self):
        # Create an empty zero image and an image with all ones
        pred = torch.zeros((1, 3, 256, 256))
        target = torch.ones((1, 3, 256, 256))
        target[:, 0, 0:32, 0:32] = 0.5

        for color_space in ["lab", "prolab"]:
            # Calculate the CDLab loss
            loss = ChromaticityDifference(color_space)(pred, target)

            # Assert that the loss is non-zero
            self.assertNotEqual(loss, 0)
