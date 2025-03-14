from __future__ import annotations
from typing import Callable, TypeAlias
from unittest import TestCase
from torch import Tensor, tensor, testing

Convert: TypeAlias = Callable[[Tensor], Tensor]


class Namespace:
    class BaseColorTest(TestCase):
        def _test(self, op: Convert, color: Tensor, ref: Tensor) -> None:
            raise NotImplementedError

    class BaseColorTestImplementation(BaseColorTest):
        wikipedia_sRGB_colors = (
            (1.000, 1.000, 1.000),
            (0.500, 0.500, 0.500),
            (0.000, 0.000, 0.000),
            (1.000, 0.000, 0.000),
            (0.750, 0.750, 0.000),
            (0.000, 0.500, 0.000),
            (0.500, 1.000, 1.000),
            (0.500, 0.500, 1.000),
            (0.750, 0.250, 0.750),
            (0.628, 0.643, 0.142),
            (0.255, 0.104, 0.918),
            (0.116, 0.675, 0.255),
            (0.941, 0.785, 0.053),
            (0.704, 0.187, 0.897),
            (0.931, 0.463, 0.316),
            (0.998, 0.974, 0.532),
            (0.099, 0.795, 0.591),
            (0.211, 0.149, 0.597),
            (0.495, 0.493, 0.721),
        )

        def test_sRGB_to_XYZ(self):
            from olimp.evaluation.cs.srgb import sRGB

            self._test(
                sRGB().to_XYZ,
                color=tensor([0.4, 0.2, 0.6]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_XYZ_to_sRGB(self):
            from olimp.evaluation.cs.srgb import sRGB

            self._test(
                sRGB().from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.4, 0.2, 0.6]),
            )

        def test_XYZ_to_ProLab(self):
            from olimp.evaluation.cs.prolab import ProLab
            from olimp.evaluation.cs import D65

            self._test(
                ProLab(D65).from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.5037, 0.1595, -0.2585]),
            )

        def test_ProLab_to_XYZ(self):
            from olimp.evaluation.cs.prolab import ProLab
            from olimp.evaluation.cs import D65

            self._test(
                ProLab(D65).to_XYZ,
                color=tensor([0.5037, 0.1595, -0.2585]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_XYZ_to_CIELAB(self):
            from olimp.evaluation.cs.cielab import CIELAB
            from olimp.evaluation.cs import D65

            self._test(
                CIELAB(illuminant_xyz=D65).from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.329039, 0.428786, -0.47156]),
            )

        def test_CIELAB_to_XYZ(self):
            from olimp.evaluation.cs.cielab import CIELAB
            from olimp.evaluation.cs import D65

            self._test(
                CIELAB(illuminant_xyz=D65).to_XYZ,
                color=tensor([0.329039, 0.428786, -0.47156]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_XYZ_to_Oklab(self):
            from olimp.evaluation.cs.oklab import Oklab

            self._test(
                Oklab().from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.44027, 0.08818, -0.13394]),
            )

        def test_Oklab_to_XYZ(self):
            from olimp.evaluation.cs.oklab import Oklab

            self._test(
                Oklab().to_XYZ,
                color=tensor([0.44027, 0.08818, -0.13394]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_XYZ_to_Opponent(self):
            from olimp.evaluation.cs.opponent import Opponent

            self._test(
                Opponent().from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.05548398, -0.05781628, 0.12142492]),
            )

        def test_Opponent_to_XYZ(self):
            from olimp.evaluation.cs.opponent import Opponent

            self._test(
                Opponent().to_XYZ,
                color=tensor([0.05548398, -0.05781628, 0.12142492]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_XYZ_to_LMS(self):
            from olimp.evaluation.cs.lms import LMS

            self._test(
                LMS().from_XYZ,
                color=tensor([0.12412, 0.07493, 0.3093]),
                ref=tensor([0.0777, 0.0734, 0.284]),
            )

        def test_LMS_to_XYZ(self):
            from olimp.evaluation.cs.lms import LMS

            self._test(
                LMS().to_XYZ,
                color=tensor([0.0777, 0.0734, 0.284]),
                ref=tensor([0.12412, 0.07493, 0.3093]),
            )

        def test_sRGB_to_HLS(self):
            from olimp.evaluation.cs.hls import HLS
            from colorsys import rgb_to_hls

            for sRGB in self.wikipedia_sRGB_colors:
                self._test(
                    HLS().from_sRGB,
                    color=tensor(sRGB),
                    ref=tensor(rgb_to_hls(*sRGB)),
                )

        def test_HLS_to_sRGB(self):
            from olimp.evaluation.cs.hls import HLS
            from colorsys import rgb_to_hls

            for sRGB in self.wikipedia_sRGB_colors:
                self._test(
                    HLS().to_sRGB,
                    color=tensor(rgb_to_hls(*sRGB)),
                    ref=tensor(sRGB),
                )

        def test_sRGB_to_HSV(self):
            from olimp.evaluation.cs.hsv import HSV
            from colorsys import rgb_to_hsv

            for sRGB in self.wikipedia_sRGB_colors:
                self._test(
                    HSV().from_sRGB,
                    color=tensor(sRGB),
                    ref=tensor(rgb_to_hsv(*sRGB)),
                )

        def test_HLS_to_HSV(self):
            from olimp.evaluation.cs.hsv import HSV
            from colorsys import rgb_to_hsv

            for sRGB in self.wikipedia_sRGB_colors:
                self._test(
                    HSV().to_sRGB,
                    color=tensor(rgb_to_hsv(*sRGB)),
                    ref=tensor(sRGB),
                )


class TestCS1D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, color: Tensor, ref: Tensor) -> None:
        exp = op(color)
        testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)


class TestCS2D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, color: Tensor, ref: Tensor) -> None:
        color = color[:, None].repeat(1, 4)
        ref = ref[:, None].repeat(1, 4)
        exp = op(color)
        testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)


class TestCS3D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, color: Tensor, ref: Tensor) -> None:
        color_repeated = color[:, None, None].repeat(1, 7, 2)
        ref = ref[:, None, None].repeat(1, 7, 2)
        exp = op(color_repeated)
        testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)
        testing.assert_close(
            color_repeated,
            color[:, None, None].repeat(1, 7, 2),
            rtol=0.0,
            atol=0.0,
        )
