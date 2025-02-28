from __future__ import annotations
from unittest import TestCase
from olimp.simulate.refraction_distortion import RefractionDistortion
import torch


class TestRetinalImageSimulator(TestCase):
    def setUp(self):
        # Set the random seed for reproducibility
        torch.manual_seed(42)

    def test_2d_convolution(self):
        I_2d = torch.rand(64, 64, dtype=torch.float)
        K_2d = torch.rand(64, 64, dtype=torch.float)
        self.distortion = RefractionDistortion()(K_2d)
        result_2d = self.distortion(I_2d)
        self.assertEqual(result_2d.shape, (64, 64))

    def test_3d_convolution(self):
        I_3d = torch.rand(3, 64, 64, dtype=torch.float)
        K_3d = torch.rand(1, 64, 64, dtype=torch.float)
        self.distortion = RefractionDistortion()(K_3d)
        result_3d = self.distortion(I_3d)
        self.assertEqual(result_3d.shape, (3, 64, 64))

    def test_4d_convolution(self):
        I_4d = torch.rand(2, 3, 64, 64, dtype=torch.float)
        K_4d = torch.rand(2, 1, 64, 64, dtype=torch.float)
        self.distortion = RefractionDistortion()(K_4d)
        result_4d = self.distortion(I_4d)
        self.assertEqual(result_4d.shape, (2, 3, 64, 64))
