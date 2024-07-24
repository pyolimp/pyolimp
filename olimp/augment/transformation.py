from __future__ import annotations
from typing import (
    Callable,
    TypeAlias,
    NamedTuple,
    Literal,
    TYPE_CHECKING,
    Any,
    Type,
)
from math import radians, hypot, sin, cos
from random import Random
from .distribution import create_distribution, DistributionParams
from .projective import Quad, projection_transform_point, calc_projection
import numpy.typing as npt
import numpy as np
import kornia

if TYPE_CHECKING:
    from .distribution import Distribution
    from torch import Tensor


U8Array: TypeAlias = npt.NDArray[np.uint8]


class ConstantDict(dict[str, int | str]):
    def pop(self, key: str, /):
        return self[key]

    def __len__(self):
        return 1


ZERO: DistributionParams = ConstantDict({"name": "constant", "value": 0})
ONE: DistributionParams = ConstantDict({"name": "constant", "value": 1})
all_transformation_classes: dict[str, Type[Transformation]] = {}


class Datum:
    image: Tensor | None
    quads: list[Quad]

    def __init__(self, image: Tensor, quad: Quad, width: int, height: int):
        """
        Image shape (N, C, H, W)
        """
        assert image.ndim == 4, image.ndim
        self.source = image
        self.quads = [quad] * image.shape[0]
        self.width = width
        self.height = height
        self.image = None

    @classmethod
    def from_tensor(cls, image: Tensor) -> Datum:
        h, w = image.shape[-2:]
        quad = (0, 0), (w, 0), (w, h), (0, h)
        return cls(image, quad=quad, width=w, height=h)


class Transformation:
    name: str

    def __init_subclass__(cls):
        if cls.name != "base":
            assert cls.name not in all_transformation_classes
            all_transformation_classes[cls.name] = cls
        super().__init_subclass__()

    def __call__(self, datum: Datum, random: Random) -> Datum:
        raise NotImplementedError


class GeometricTransform(Transformation):
    name = "base"

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        raise NotImplementedError

    def __call__(self, datum: Datum, random: Random) -> Datum:
        datum.quads = [
            self.new_quad(quad, datum, random) for quad in datum.quads
        ]
        return datum


class Projective1ptTransformation(GeometricTransform):
    """
    Shifts one point of the quadrangle in random direction.
    """

    name = "projective1pt"

    def __init__(
        self,
        x_distribution: DistributionParams,
        y_distribution: DistributionParams,
    ):
        """
        :param x_distribution: `create_distribution` arguments, dict
        :param y_distribution: `create_distribution` arguments, dict
        """
        self._x_distribution = create_distribution(x_distribution)
        self._y_distribution = create_distribution(y_distribution)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]  # guess the scale
        size = max(abs(x2 - x1), abs(y2 - y1))
        shift_x = self._x_distribution(random) * size
        shift_y = self._y_distribution(random) * size
        out_quad = list(quad)
        random_point_idx = random.randint(0, 3)
        point = out_quad[random_point_idx]
        out_quad[random_point_idx] = (point[0] + shift_x, point[1] + shift_y)

        return tuple(out_quad)


class Projective4ptTransformation(GeometricTransform):
    """
    Shifts four point of the quadrangle in random direction.
    """

    name = "projective4pt"

    def __init__(
        self,
        x_distribution: DistributionParams,
        y_distribution: DistributionParams,
    ):
        """
        :param x_distribution: `create_distribution` arguments, dict
        :param y_distribution: `create_distribution` arguments, dict
        """
        self._x_distribution = create_distribution(x_distribution)
        self._y_distribution = create_distribution(y_distribution)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]  # guess the scale
        size = max(abs(x2 - x1), abs(y2 - y1))
        return tuple(
            [
                (
                    point[0] + self._x_distribution(random) * size,
                    point[1] + self._y_distribution(random) * size,
                )
                for point in quad
            ]
        )


class Flip(GeometricTransform):
    """
    Flips the quadrangle vertically or horizontally.
    Only changes points order, that is, visually the quadrangle doesn't
    change, but its visualization does.

    For diagonal names see: https://en.wikipedia.org/wiki/Main_diagonal
    """

    name = "flip"

    def __init__(
        self,
        direction: Literal[
            "horizontal", "vertical", "primary_diagonal", "secondary_diagonal"
        ] = "horizontal",
    ):
        self._direction = getattr(self, direction)

    @staticmethod
    def horizontal(q: Quad) -> Quad:
        """.. image:: _static/transformations/flip_horizontal.svg"""
        return (q[1], q[0], q[3], q[2])

    @staticmethod
    def vertical(q: Quad) -> Quad:
        """.. image:: _static/transformations/flip_vertical.svg"""
        return (q[2], q[3], q[0], q[1])

    @staticmethod
    def primary_diagonal(q: Quad) -> Quad:
        """.. image:: _static/transformations/flip_primary_diagonal.svg"""
        return (q[0], q[3], q[2], q[1])

    @staticmethod
    def secondary_diagonal(q: Quad) -> Quad:
        """.. image:: _static/transformations/flip_secondary_diagonal.svg"""
        return (q[2], q[1], q[0], q[3])

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        return self._direction(quad)


class PaddingsAddition(GeometricTransform):
    """
    Adds random padding to the quadrangle sides.

    .. image:: _static/transformations/paddings_addition.svg
    """

    name = "paddings_addition"

    def __init__(
        self,
        top: DistributionParams = ZERO,
        right: DistributionParams = ZERO,
        bottom: DistributionParams = ZERO,
        left: DistributionParams = ZERO,
    ):
        self._top = create_distribution(top)
        self._right = create_distribution(right)
        self._bottom = create_distribution(bottom)
        self._left = create_distribution(left)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]
        width, height = abs(x2 - x1), abs(y2 - y1)

        shift_top = self._top(random) * height
        shift_right = self._right(random) * width
        shift_bottom = self._bottom(random) * height
        shift_left = self._left(random) * width

        return (
            (quad[0][0] + shift_left, quad[0][1] + shift_top),
            (quad[1][0] + shift_right, quad[1][1] + shift_top),
            (quad[2][0] + shift_right, quad[2][1] + shift_bottom),
            (quad[3][0] + shift_left, quad[3][1] + shift_bottom),
        )


class ProjectivePaddingsAddition(PaddingsAddition):
    """
    Same as `PaddingsAddition`, but addition respects original projective
    transformation.

    .. image:: _static/transformations/projective_paddings_addition.svg
    """

    name = "projective_paddings_addition"

    def __init__(
        self,
        top: DistributionParams = ZERO,
        right: DistributionParams = ZERO,
        bottom: DistributionParams = ZERO,
        left: DistributionParams = ZERO,
    ):
        self._top = create_distribution(top)
        self._right = create_distribution(right)
        self._bottom = create_distribution(bottom)
        self._left = create_distribution(left)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        width, height = datum.width, datum.height

        shift_top = self._top(random) * height
        shift_right = self._right(random) * width
        shift_bottom = self._bottom(random) * height
        shift_left = self._left(random) * width

        rect = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
        m = calc_projection(rect, quad)

        return (
            projection_transform_point((shift_left, shift_top), m),
            projection_transform_point((width + shift_right, shift_top), m),
            projection_transform_point(
                (width + shift_right, height + shift_bottom), m
            ),
            projection_transform_point((shift_left, height + shift_bottom), m),
        )


class Rotate(GeometricTransform):
    """
    Rotates the quadrangle around its center.

    .. image:: _static/transformations/rotate.svg
    """

    name = "rotate"

    def __init__(self, angle: DistributionParams):
        """
        :param angle: `create_distribution` arguments, dict
        """
        self._angle = create_distribution(angle)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        angle = radians(self._angle(random))
        return self._rotate_center(quad, angle)

    @staticmethod
    def _get_center(points: Quad):
        sx = sy = sL = 0
        for i, (x1, y1) in enumerate(points):
            x0, y0 = points[i - 1]
            L = hypot(x1 - x0, y1 - y0)
            sx += (x0 + x1) * 0.5 * L
            sy += (y0 + y1) * 0.5 * L
            sL += L
        return sx / sL, sy / sL

    @classmethod
    def _rotate_center(cls, quad: Quad, angle: float) -> Quad:
        ox, oy = cls._get_center(quad)
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        return tuple(
            [
                (
                    ox + cos_angle * (x - ox) - sin_angle * (y - oy),
                    oy + sin_angle * (x - ox) + cos_angle * (y - oy),
                )
                for x, y in quad
            ]
        )


class ProjectiveShift(GeometricTransform):
    """
    Projectively shifts the quadrangle.

    .. image:: _static/transformations/projective_shift.svg
    """

    name = "projective_shift"

    def __init__(
        self,
        x: DistributionParams = ZERO,
        y: DistributionParams = ZERO,
    ):
        """
        :param x: `create_distribution` arguments, dict
        :param y: `create_distribution` arguments, dict
        """
        self._x = create_distribution(x)
        self._y = create_distribution(y)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        width, height = datum.width, datum.height
        x_shift = self._x(random) * width
        y_shift = self._y(random) * height
        rect = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
        m = calc_projection(rect, quad)

        return (
            projection_transform_point((x_shift, y_shift), m),
            projection_transform_point((width + x_shift, y_shift), m),
            projection_transform_point((width + x_shift, height + y_shift), m),
            projection_transform_point((x_shift, height + y_shift), m),
        )


class Scale(GeometricTransform):
    """
    Scales the quadrangle to the factor specified in the `distribution`.

    .. image:: _static/transformations/scale.svg
    """

    name = "scale"

    def __init__(self, factor: DistributionParams):
        """
        :param factor: `create_distribution` arguments, dict
        """
        self._factor = create_distribution(factor)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        centre_x = sum([quad[i][0] for i in range(len(quad))]) / len(quad)
        centre_y = sum([quad[i][1] for i in range(len(quad))]) / len(quad)
        scale = self._factor(random)

        out_quad = list(quad)
        for i in range(len(out_quad)):
            out_quad[i] = (
                centre_x + scale * (out_quad[i][0] - centre_x),
                centre_y + scale * (out_quad[i][1] - centre_y),
            )

        return tuple(out_quad)


class Rasterize(Transformation):
    """
    Rasterizes the image from quadrangle using projective transform and
    the size specified in `Descriptor`.
    """

    name = "rasterize"

    def __call__(self, datum: Datum, _random: Random) -> Datum:
        rect = self.rect(datum)
        mat_pys = [calc_projection(rect, quad) for quad in datum.quads]
        print("mat_pys", mat_pys)
        import torch

        mat = torch.tensor(mat_pys, dtype=torch.float32)
        datum.image = kornia.geometry.homography_warp(
            datum.source,
            mat,
            (datum.height, datum.width),
            normalized_coordinates=False,
            normalized_homography=False,
            align_corners=True,
        )
        return datum

    def rect(self, datum: Datum) -> Quad:
        return (
            (0.0, 0.0),
            (datum.width, 0.0),
            (datum.width, datum.height),
            (0.0, datum.height),
        )


class Noising(Transformation):
    """
    Adds normal noise to the image `numpy.random.RandomState.normal`.

    .. image:: _static/transformations/noising.svg
    """

    name = "noising"

    def __init__(self, mean: DistributionParams, std: DistributionParams):
        """
        see `kornia.augmentation.RandomGaussianNoise`
        """
        self._mean = create_distribution(mean)
        self._std = create_distribution(std)

    def __call__(self, datum: Datum, random: Random):
        kornia.augmentation.RandomGaussianNoise(
            mean=self._mean(random),
            std=self._std(random),
            same_on_batch=False,
            p=1.0,
        )
        return datum


class Brightness(Transformation):
    name = "brightness"

    def __init__(self, factor: DistributionParams):
        """
        see `kornia.enhance.adjust_brightness`
        """
        self._factor = create_distribution(factor)

    def __call__(self, datum: Datum, random: Random):
        assert datum.image
        datum.image = kornia.enhance.adjust_brightness(
            datum.image, factor=self._factor(random)
        )
        return datum


class Contrast(Transformation):
    name = "contrast"

    def __init__(self, factor: DistributionParams):
        """
        see `kornia.enhance.adjust_contrast`
        """
        self._factor = create_distribution(factor)

    def __call__(self, datum: Datum, random: Random):
        assert datum.image
        datum.image = kornia.enhance.adjust_contrast(
            datum.image, factor=self._factor(random)
        )
        return datum


class Gamma(Transformation):
    name = "gamma"

    def __init__(
        self, gamma: DistributionParams, gain: DistributionParams = ONE
    ):
        """
        see `kornia.enhance.adjust_gamma`
        """
        self._gamma = create_distribution(gamma)
        self._gain = create_distribution(gain)

    def __call__(self, datum: Datum, random: Random):
        assert datum.image
        datum.image = kornia.enhance.adjust_gamma(
            datum.image, gamma=self._gamma(random), gain=self._gain(random)
        )
        return datum


class Log(Transformation):
    name = "log"

    def __init__(self, gain: DistributionParams):
        """
        see `kornia.enhance.adjust_log`
        """
        self._gain = create_distribution(gain)

    def __call__(self, datum: Datum, random: Random):
        assert datum.image
        datum.image = kornia.enhance.adjust_log(
            datum.image, gain=self._gain(random)
        )
        return datum


class Sharpness(Transformation):
    name = "sharpness"

    def __init__(self, factor: DistributionParams):
        """
        see `kornia.enhance.sharpness`
        """
        self._factor = create_distribution(factor)

    def __call__(self, datum: Datum, random: Random):
        assert datum.image
        datum.image = kornia.enhance.sharpness(
            datum.image, factor=self._factor(random)
        )
        return datum


def create(name: str, kwargs: Any) -> Transformation:
    if name not in all_transformation_classes:
        raise Exception(
            f"Unknown transformation name `{name}`, "
            f"available names are: {sorted(all_transformation_classes)}"
        )
    cls = all_transformation_classes[name]

    try:
        return cls(**kwargs)
    except Exception as e:
        raise ValueError(
            f"Exception in {cls} ({name}) for arguments {kwargs}"
        ) from e
