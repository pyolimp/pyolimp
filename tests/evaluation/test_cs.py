from __future__ import annotations
from typing import Callable, TypeAlias
from unittest import TestCase
from torch import Tensor as T, tensor, testing, hstack, dstack

Convert: TypeAlias = Callable[[T], T]


class Namespace:
    class BaseColorTest(TestCase):
        def _test(
            self, op: Convert, col0: T, col1: T, ref0: T, ref1: T
        ) -> None:
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
                col0=tensor([0.4, 0.2, 0.6]),
                col1=tensor([0.0, 0.1, 0.9]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_XYZ_to_sRGB(self):
            from olimp.evaluation.cs.srgb import sRGB

            self._test(
                sRGB().from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.4, 0.2, 0.6]),
                ref1=tensor([0.0, 0.1, 0.9]),
            )

        def test_XYZ_to_ProLab(self):
            from olimp.evaluation.cs.prolab import ProLab
            from olimp.evaluation.cs import D65

            self._test(
                ProLab(D65).from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.5037, 0.1595, -0.2585]),
                ref1=tensor([0.6278956532, 0.16390889883, -0.5866720676]),
            )

        def test_ProLab_to_XYZ(self):
            from olimp.evaluation.cs.prolab import ProLab
            from olimp.evaluation.cs import D65

            self._test(
                ProLab(D65).to_XYZ,
                col0=tensor([0.5037, 0.1595, -0.2585]),
                col1=tensor([0.6278956532, 0.16390889883, -0.5866720676]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_XYZ_to_CIELAB(self):
            from olimp.evaluation.cs.cielab import CIELAB
            from olimp.evaluation.cs import D65

            self._test(
                CIELAB(illuminant_xyz=D65).from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.329039, 0.428786, -0.47156]),
                ref1=tensor([0.3039984107, 0.6757221222, -0.9658879638]),
            )

        def test_CIELAB_to_XYZ(self):
            from olimp.evaluation.cs.cielab import CIELAB
            from olimp.evaluation.cs import D65

            self._test(
                CIELAB(illuminant_xyz=D65).to_XYZ,
                col0=tensor([0.329039, 0.428786, -0.47156]),
                col1=tensor([0.3039984107, 0.6757221222, -0.9658879638]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_XYZ_to_Oklab(self):
            from olimp.evaluation.cs.oklab import Oklab

            self._test(
                Oklab().from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.44027, 0.08818, -0.13394]),
                ref1=tensor([0.42955389619, -0.02839118242, -0.2795809805]),
            )

        def test_Oklab_to_XYZ(self):
            from olimp.evaluation.cs.oklab import Oklab

            self._test(
                Oklab().to_XYZ,
                col0=tensor([0.44027, 0.08818, -0.13394]),
                col1=tensor([0.42955389619, -0.02839118242, -0.2795809805]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_XYZ_to_Opponent(self):
            from olimp.evaluation.cs.opponent import Opponent

            self._test(
                Opponent().from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.05548398, -0.05781628, 0.12142492]),
                ref1=tensor([0.006525486242, -0.10455235094, 0.35025477409]),
            )

        def test_Opponent_to_XYZ(self):
            from olimp.evaluation.cs.opponent import Opponent

            self._test(
                Opponent().to_XYZ,
                col0=tensor([0.05548398, -0.05781628, 0.12142492]),
                col1=tensor([0.006525486242, -0.10455235094, 0.35025477409]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_XYZ_to_LMS(self):
            from olimp.evaluation.cs.lms import LMS

            self._test(
                LMS().from_XYZ,
                col0=tensor([0.12412, 0.07493, 0.3093]),
                col1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
                ref0=tensor([0.0777, 0.0734, 0.284]),
                ref1=tensor([0.04302080348, 0.07586731016, 0.6881836653]),
            )

        def test_LMS_to_XYZ(self):
            from olimp.evaluation.cs.lms import LMS

            self._test(
                LMS().to_XYZ,
                col0=tensor([0.0777, 0.0734, 0.284]),
                col1=tensor([0.04302080348, 0.07586731016, 0.6881836653]),
                ref0=tensor([0.12412, 0.07493, 0.3093]),
                ref1=tensor([0.14566263556, 0.06399934, 0.7494758367]),
            )

        def test_sRGB_to_HLS(self):
            from olimp.evaluation.cs.hls import HLS
            from colorsys import rgb_to_hls

            col1 = tensor(self.wikipedia_sRGB_colors[0])
            ref1 = tensor(rgb_to_hls(*col1))

            for sRGB in self.wikipedia_sRGB_colors[1:]:
                self._test(
                    HLS().from_sRGB,
                    col0=tensor(sRGB),
                    col1=col1,
                    ref0=tensor(rgb_to_hls(*sRGB)),
                    ref1=ref1,
                )

        def test_HLS_to_sRGB(self):
            from olimp.evaluation.cs.hls import HLS
            from colorsys import rgb_to_hls

            ref1 = tensor(self.wikipedia_sRGB_colors[0])
            col1 = tensor(rgb_to_hls(*ref1))

            for sRGB in self.wikipedia_sRGB_colors[1:]:
                self._test(
                    HLS().to_sRGB,
                    col0=tensor(rgb_to_hls(*sRGB)),
                    col1=col1,
                    ref0=tensor(sRGB),
                    ref1=ref1,
                )

        def test_sRGB_to_HSV(self):
            from olimp.evaluation.cs.hsv import HSV
            from colorsys import rgb_to_hsv

            col1 = tensor(self.wikipedia_sRGB_colors[0])
            ref1 = tensor(rgb_to_hsv(*col1))
            for sRGB in self.wikipedia_sRGB_colors[1:]:
                self._test(
                    HSV().from_sRGB,
                    col0=tensor(sRGB),
                    col1=col1,
                    ref0=tensor(rgb_to_hsv(*sRGB)),
                    ref1=ref1,
                )

        def test_HLS_to_HSV(self):
            from olimp.evaluation.cs.hsv import HSV
            from colorsys import rgb_to_hsv

            ref1 = tensor(self.wikipedia_sRGB_colors[0])
            col1 = tensor(rgb_to_hsv(*ref1))

            for sRGB in self.wikipedia_sRGB_colors[1:]:
                self._test(
                    HSV().to_sRGB,
                    col0=tensor(rgb_to_hsv(*sRGB)),
                    col1=col1,
                    ref0=tensor(sRGB),
                    ref1=ref1,
                )


class TestCS1D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, col0: T, col1: T, ref0: T, ref1: T) -> None:
        for color, ref in (col0, ref0), (col1, ref1):
            exp = op(color)
            testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)


class TestCS2D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, col0: T, col1: T, ref0: T, ref1: T) -> None:
        color = hstack(
            (col0[:, None].repeat(1, 2), col1[:, None].repeat(1, 2))
        )
        ref = hstack((ref0[:, None].repeat(1, 2), ref1[:, None].repeat(1, 2)))
        exp = op(color)
        testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)


class TestCS3D(Namespace.BaseColorTestImplementation):
    def _test(self, op: Convert, col0: T, col1: T, ref0: T, ref1: T) -> None:
        color_repeated = dstack(
            (
                col0[:, None, None].repeat(1, 7, 2),
                col1[:, None, None].repeat(1, 7, 2),
            )
        )
        ref = dstack(
            (
                ref0[:, None, None].repeat(1, 7, 2),
                ref1[:, None, None].repeat(1, 7, 2),
            )
        )
        exp = op(color_repeated)
        testing.assert_close(exp, ref, rtol=1e-5, atol=1e-4)
