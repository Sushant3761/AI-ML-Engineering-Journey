"""Matrix operations implemented with Python loops and lists."""

from __future__ import annotations

from numbers import Real
from typing import Sequence

from ._validation import Matrix, Vector, as_matrix, as_vector, finite_scalar, same_shape


def add(left: Sequence[Sequence[Real]], right: Sequence[Sequence[Real]]) -> Matrix:
    """Return the element-wise sum of equal-shaped matrices."""
    a, b = as_matrix(left, name="left"), as_matrix(right, name="right")
    same_shape(a, b, names=("left", "right"))
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def subtract(left: Sequence[Sequence[Real]], right: Sequence[Sequence[Real]]) -> Matrix:
    """Return the element-wise difference ``left - right``."""
    a, b = as_matrix(left, name="left"), as_matrix(right, name="right")
    same_shape(a, b, names=("left", "right"))
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def scale(matrix: Sequence[Sequence[Real]], scalar: Real) -> Matrix:
    """Multiply every matrix entry by a finite scalar."""
    values, factor = as_matrix(matrix), finite_scalar(scalar)
    return [[factor * value for value in row] for row in values]


def multiply(left: Sequence[Sequence[Real]], right: Sequence[Sequence[Real]]) -> Matrix:
    """Return the matrix product ``left @ right``."""
    a, b = as_matrix(left, name="left"), as_matrix(right, name="right")
    if len(a[0]) != len(b):
        raise ValueError(f"incompatible matrix shapes: {(len(a), len(a[0]))} and {(len(b), len(b[0]))}")
    columns = transpose(b)
    return [[sum(x * y for x, y in zip(row, column)) for column in columns] for row in a]


def transpose(matrix: Sequence[Sequence[Real]]) -> Matrix:
    """Return the transpose of a rectangular matrix."""
    values = as_matrix(matrix)
    return [[values[row][column] for row in range(len(values))] for column in range(len(values[0]))]


def identity(size: int) -> Matrix:
    """Return an ``size`` by ``size`` identity matrix."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def multiply_vector(matrix: Sequence[Sequence[Real]], vector: Sequence[Real]) -> Vector:
    """Return the product of a matrix and a compatible column vector."""
    values, items = as_matrix(matrix), as_vector(vector, name="vector")
    if len(values[0]) != len(items):
        raise ValueError(f"matrix columns ({len(values[0])}) must match vector length ({len(items)})")
    return [sum(entry * value for entry, value in zip(row, items)) for row in values]


def shape(matrix: Sequence[Sequence[Real]]) -> tuple[int, int]:
    """Return the row and column count after validating the matrix."""
    values = as_matrix(matrix)
    return len(values), len(values[0])
