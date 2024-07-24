from __future__ import annotations
from typing import NamedTuple, TypeAlias, Sequence


Point: TypeAlias = tuple[float, float]
Quad: TypeAlias = tuple[Point, Point, Point, Point]


Matrix3x3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _multiply_matrix3x3(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3:
    return tuple(
        [
            tuple(
                [
                    sum(a[row][i] * b[i][col] for i in range(3))
                    for col in range(3)
                ]
            )
            for row in range(3)
        ]
    )


def _inverse_matrix_3x3(inp: Matrix3x3) -> Matrix3x3:
    det = (
        inp[0][0] * inp[1][1] * inp[2][2]
        + inp[0][1] * inp[1][2] * inp[2][0]
        + inp[0][2] * inp[1][0] * inp[2][1]
        - inp[0][0] * inp[1][2] * inp[2][1]
        - inp[0][1] * inp[1][0] * inp[2][2]
        - inp[0][2] * inp[1][1] * inp[2][0]
    )
    if not det:
        raise ZeroDivisionError("singular matrix")

    return (
        (
            (inp[1][1] * inp[2][2] - inp[1][2] * inp[2][1]) / det,
            -(inp[0][1] * inp[2][2] - inp[0][2] * inp[2][1]) / det,
            (inp[0][1] * inp[1][2] - inp[0][2] * inp[1][1]) / det,
        ),
        (
            -(inp[1][0] * inp[2][2] - inp[1][2] * inp[2][0]) / det,
            (inp[0][0] * inp[2][2] - inp[0][2] * inp[2][0]) / det,
            -(inp[0][0] * inp[1][2] - inp[0][2] * inp[1][0]) / det,
        ),
        (
            (inp[1][0] * inp[2][1] - inp[1][1] * inp[2][0]) / det,
            -(inp[0][0] * inp[2][1] - inp[0][1] * inp[2][0]) / det,
            (inp[0][0] * inp[1][1] - inp[0][1] * inp[1][0]) / det,
        ),
    )


def _solve2lin(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> tuple[float, float]:
    d = a * e - b * d
    return (
        (c * e - b * f) / d,
        (a * f - c * d) / d,
    )


# Calculate the projection matrix given 4 points
def _calc_projection1(ps: Quad) -> Matrix3x3:
    td = _solve2lin(
        ps[3][0] - ps[2][0],
        ps[3][0] - ps[1][0],
        ps[3][0] - ps[0][0],
        ps[3][1] - ps[2][1],
        ps[3][1] - ps[1][1],
        ps[3][1] - ps[0][1],
    )

    row_2 = (td[0] - 1.0, td[1] - 1.0, 1.0)
    return (
        (
            ps[2][0] * row_2[0] + ps[2][0] - ps[0][0],
            ps[1][0] * row_2[1] + ps[1][0] - ps[0][0],
            ps[0][0],
        ),
        (
            ps[2][1] * row_2[0] + ps[2][1] - ps[0][1],
            ps[1][1] * row_2[1] + ps[1][1] - ps[0][1],
            ps[0][1],
        ),
        row_2,
    )


def calc_projection(src_coords: Quad, dst_coords: Quad) -> Matrix3x3:
    return _multiply_matrix3x3(
        _calc_projection1(dst_coords),
        _inverse_matrix_3x3(_calc_projection1(src_coords)),
    )


def projection_transform_point(
    src_point: Point, projective_matrix: Matrix3x3
) -> Point:
    div = (
        projective_matrix[2][0] * src_point[0]
        + projective_matrix[2][1] * src_point[1]
        + projective_matrix[2][2]
    )
    if not div:
        raise ZeroDivisionError("singular matrix")

    return tuple(
        [
            (
                projective_matrix[i][0] * src_point[0]
                + projective_matrix[i][1] * src_point[1]
                + projective_matrix[i][2]
            )
            / div
            for i in range(2)
        ]
    )


if __name__ == "__main__":
    I = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    assert projection_transform_point((0.5, 0.5), I) == (0.5, 0.5)

    p1 = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    p2 = (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)
    assert calc_projection(p1, p1) == I
    assert calc_projection(p2, p1) == (
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 1),
    )
