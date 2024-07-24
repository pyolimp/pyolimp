from __future__ import annotations
from typing import Callable
from math import sqrt, erf
from scipy.special import erfinv as _erfinv


SQTR2 = sqrt(2.0)


def _cdf(mu: float, sigma: float, x: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * SQTR2)))


def _cdf_inv(mu: float, sigma: float, p: float) -> float:
    assert 0.0 < p < 1.0, p
    return mu + sigma * SQTR2 * _erfinv((p + p) - 1.0)


def truncnorm(
    mu: float, sigma: float, a: float, b: float
) -> Callable[[float], float]:
    cdf_a = _cdf(mu, sigma, a)
    p_coef = _cdf(mu, sigma, b) - cdf_a

    def truncnorm(p: float, _cdf_inv=_cdf_inv) -> float:
        return _cdf_inv(mu, sigma, cdf_a + p * p_coef)

    return truncnorm
