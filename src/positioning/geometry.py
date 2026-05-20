"""Small 3D geometry primitives used by the TDOA solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale: float) -> "Vec3":
        return self * scale

    def __truediv__(self, scale: float) -> "Vec3":
        return Vec3(self.x / scale, self.y / scale, self.z / scale)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def norm(v: Vec3) -> float:
    return sqrt(dot(v, v))


def distance(a: Vec3, b: Vec3) -> float:
    return norm(a - b)


def position_error(a: Vec3, b: Vec3) -> float:
    return distance(a, b)


def centroid(points: Iterable[Vec3]) -> Vec3:
    items = list(points)
    if not items:
        raise ValueError("centroid requires at least one point")
    total = Vec3()
    for point in items:
        total = total + point
    return total / float(len(items))


def solve3x3(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Vec3:
    """Solve a 3x3 linear system using Gauss-Jordan elimination."""
    if len(matrix) != 3 or len(rhs) != 3:
        raise ValueError("solve3x3 expects a 3x3 matrix and 3-element rhs")

    aug = [
        [float(matrix[row][0]), float(matrix[row][1]), float(matrix[row][2]), float(rhs[row])]
        for row in range(3)
    ]

    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("degenerate geometry or insufficient data for localization")

        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        diag = aug[col][col]
        for j in range(col, 4):
            aug[col][j] /= diag

        for row in range(3):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, 4):
                aug[row][j] -= factor * aug[col][j]

    return Vec3(aug[0][3], aug[1][3], aug[2][3])


def try_solve3x3(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Vec3 | None:
    try:
        result = solve3x3(matrix, rhs)
    except ValueError:
        return None
    if not all(value == value and abs(value) != float("inf") for value in result.as_tuple()):
        return None
    return result
