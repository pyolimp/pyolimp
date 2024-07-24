from __future__ import annotations
from unittest import TestCase
from olimp.simulat import RetinalImageSimulator
import torch


class TestRetinalImageSimulator(TestCase):
    def setUp(self):
        # Set the random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the simulator
        self.simulator = RetinalImageSimulator()

    def test_2d_convolution_2d(self):
        I_2d = torch.rand(64, 64, dtype=torch.float)
        K_2d = torch.rand(64, 64, dtype=torch.float)
        result_2d = self.simulator.fast_conv(I_2d, K_2d, return_2d=True)
        self.assertEqual(result_2d.shape, (64, 64))

    def test_2d_convolution(self):
        I_2d = torch.rand(64, 64, dtype=torch.float)
        K_2d = torch.rand(64, 64, dtype=torch.float)
        result_2d = self.simulator.fast_conv(I_2d, K_2d)
        self.assertEqual(result_2d.shape, (1, 1, 64, 64))

    def test_3d_convolution(self):
        I_3d = torch.rand(3, 64, 64, dtype=torch.float)
        K_3d = torch.rand(1, 64, 64, dtype=torch.float)
        result_3d = self.simulator.fast_conv(I_3d, K_3d)
        self.assertEqual(result_3d.shape, (1, 3, 64, 64))

    def test_4d_convolution(self):
        I_4d = torch.rand(2, 3, 64, 64, dtype=torch.float)
        K_4d = torch.rand(2, 1, 64, 64, dtype=torch.float)
        result_4d = self.simulator.fast_conv(I_4d, K_4d)
        self.assertEqual(result_4d.shape, (2, 3, 64, 64))
