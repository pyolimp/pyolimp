from __future__ import annotations
from unittest import TestCase
from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
import torch


class ShapeTest(TestCase):
    def test_shape_3d(self):
        I_3d = torch.rand(1, 3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Vienot", blindness_type="PROTAN")
        res = distortion(I_3d)
        self.assertEqual(res.shape, (1, 3, 64, 64))

    def test_shape_4d(self):
        I_4d = torch.rand(2, 3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Vienot", blindness_type="PROTAN")
        res = distortion(I_4d)
        self.assertEqual(res.shape, (2, 3, 64, 64))


class RangeTest(TestCase):
    def test_protan_vienot_range(self):
        image = torch.rand(3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Vienot", blindness_type="PROTAN")
        res = distortion(image)
        self.assertTrue(res.min() >= 0 and res.max() <= 1)

    def test_protan_huang_range(self):
        image = torch.rand(3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Huang", blindness_type="PROTAN")
        res = distortion(image)
        self.assertTrue(res.min() >= 0 and res.max() <= 1)

    def test_deutan_vienot_range(self):
        image = torch.rand(3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Vienot", blindness_type="DEUTAN")
        res = distortion(image)
        self.assertTrue(res.min() >= 0 and res.max() <= 1)

    def test_deutan_huang_range(self):
        image = torch.rand(3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Huang", blindness_type="DEUTAN")
        res = distortion(image)
        self.assertTrue(res.min() >= 0 and res.max() <= 1)

    def test_rg_farup_range(self):
        image = torch.rand(3, 64, 64, dtype=torch.float)
        distortion = ColorBlindnessDistortion(sim_type="Farup", blindness_type="rg")
        res = distortion(image)
        self.assertTrue(res.min() >= 0 and res.max() <= 1)
