"""Shared input validation for the pure-Python linear algebra routines."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Sequence, TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[list[float]]


def as_vector(values: Sequence[Real], *, name: str = "vector", allow_empty: bool = False) -> Vector:
    """Copy a one-dimensional finite numeric sequence into a list of floats."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        items = list(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in items):
        raise TypeError(f"{name} must contain real numbers")
    result = [float(value) for value in items]
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def as_matrix(values: Sequence[Sequence[Real]], *, name: str = "matrix") -> Matrix:
    """Copy a rectangular, non-empty matrix into lists of floats."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a two-dimensional numeric sequence")
    try:
        rows = [as_vector(row, name=f"{name} row {index}") for index, row in enumerate(values)]
    except TypeError:
        raise
    if not rows:
        raise ValueError(f"{name} must not be empty (must have at least one row)")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def same_length(left: Sequence[object], right: Sequence[object], *, names: tuple[str, str]) -> None:
    """Require two vectors to have equal dimensions."""
    if len(left) != len(right):
        raise ValueError(f"{names[0]} and {names[1]} dimensions differ: {len(left)} != {len(right)}")


def same_shape(left: Matrix, right: Matrix, *, names: tuple[str, str]) -> None:
    """Require two matrices to have equal shapes."""
    left_shape = (len(left), len(left[0]))
    right_shape = (len(right), len(right[0]))
    if left_shape != right_shape:
        raise ValueError(f"{names[0]} and {names[1]} shapes differ: {left_shape} != {right_shape}")


def finite_scalar(value: Real, *, name: str = "scalar") -> float:
    """Convert a real scalar to float and reject non-finite values."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
