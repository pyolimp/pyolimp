from __future__ import annotations
from typing import TYPE_CHECKING
from typing import TypedDict

if TYPE_CHECKING:
    from random import Random
    from typing import (
        Callable,
        Literal,
        NotRequired,
        TypeAlias,
    )

    Distribution: TypeAlias = Callable[[Random], float]


class UniformParams(TypedDict):
    name: Literal["uniform"]
    a: NotRequired[float]
    b: NotRequired[float]


class GaussParams(TypedDict):
    name: Literal["gauss"]
    a: NotRequired[float]
    b: NotRequired[float]
    mu: NotRequired[float]
    sigma: NotRequired[float]
    delta: NotRequired[float]


class ConstantParams(TypedDict):
    name: Literal["constant"]
    value: NotRequired[float]


class RandrangeParams(TypedDict):
    name: Literal["randrange"]
    start: NotRequired[int]
    stop: NotRequired[int]


DistributionParams: TypeAlias = (
    UniformParams | GaussParams | ConstantParams | RandrangeParams
)


def create_distribution(
    kwargs: DistributionParams,
) -> Distribution:
    """
    .. list-table:: Available Distributions
       :widths: 5 10 10
       :header-rows: 1

       * - Name
         - Parameters
         - Distribution
       * - uniform
         - a=0, b=0.5
         - .. image:: _static/transformations/uniform_000_050.svg
              :width: 75%
       * - uniform
         - a=0.75, b=0.75
         - .. image:: _static/transformations/uniform_075_075.svg
              :width: 75%
       * - gauss
         - mu=0.0, sigma=0.75, delta=1.0
         - .. image:: _static/transformations/gauss_000_075_100.svg
              :width: 75%
       * - gauss
         - mu=0.4, sigma=0.3, delta=1.0
         - .. image:: _static/transformations/gauss_040_030_100.svg
              :width: 75%
       * - gauss
         - mu=0.0, sigma=0.5, a=0.0, b=1.0
         - .. image:: _static/transformations/gauss_000_050_000_100.svg
              :width: 75%
       * - constant
         - value=0.25
         - .. image:: _static/transformations/constant_025.svg
              :width: 75%
       * - randrange
         - start=-1, stop=2
         - .. image:: _static/transformations/randrange_-1_2.svg
              :width: 75%

    :param name: distribution name in ['uniform', 'gauss', 'constant', 'randrange']
    """
    match kwargs["name"]:
        case "uniform":

            def _uniform(
                random: Random,
                a: float = kwargs.pop("a"),
                b: float = kwargs.pop("b"),
            ) -> float:
                return random.uniform(a, b)

            ret = _uniform

        case "gauss":
            from .truncnorm import truncnorm

            mu = kwargs.pop("mu", 0.0)
            if "delta" in kwargs:
                assert "a" not in kwargs and "b" not in kwargs
                delta = kwargs.pop("delta")
                a, b = mu - delta, mu + delta
            else:
                a, b = kwargs.pop("a"), kwargs.pop("b")
            f = truncnorm(mu, kwargs.pop("sigma", 1.0), a, b)

            def _gauss(
                random: Random, f: Callable[[float], float] = f
            ) -> float:
                return f(random.random())

            ret = _gauss

        case "constant":
            value = kwargs.pop("value")

            def _constant(random: Random, value: float = value) -> float:
                return value

            ret = _constant

        case "randrange":

            def _randrange(
                random: Random,
                start: int = kwargs.pop("start"),
                stop: int = kwargs.pop("stop"),
            ):
                return random.randrange(start, stop)

            ret = _randrange
        case _:
            raise ValueError(f"Unsupported distribution: {kwargs['name']}")
    if len(kwargs) != 1:
        raise ValueError(f"Can't create distribution from kwargs: {kwargs}")
    return ret
