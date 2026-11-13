"""Vector operations implemented with Python sequences."""

from __future__ import annotations

from math import sqrt
from numbers import Real
from typing import Sequence

from ._validation import Vector, as_vector, finite_scalar, same_length


def add(left: Sequence[Real], right: Sequence[Real]) -> Vector:
    """Return the component-wise sum of equal-dimensional vectors."""
    a, b = as_vector(left, name="left"), as_vector(right, name="right")
    same_length(a, b, names=("left", "right"))
    return [x + y for x, y in zip(a, b)]


def subtract(left: Sequence[Real], right: Sequence[Real]) -> Vector:
    """Return the component-wise difference ``left - right``."""
    a, b = as_vector(left, name="left"), as_vector(right, name="right")
    same_length(a, b, names=("left", "right"))
    return [x - y for x, y in zip(a, b)]


def scale(vector: Sequence[Real], scalar: Real) -> Vector:
    """Multiply every component by a finite scalar."""
    values, factor = as_vector(vector), finite_scalar(scalar)
    return [factor * value for value in values]


def dot(left: Sequence[Real], right: Sequence[Real]) -> float:
    """Return the Euclidean dot product of equal-dimensional vectors."""
    a, b = as_vector(left, name="left"), as_vector(right, name="right")
    same_length(a, b, names=("left", "right"))
    return sum(x * y for x, y in zip(a, b))


def magnitude(vector: Sequence[Real]) -> float:
    """Return the Euclidean (L2) norm."""
    return sqrt(dot(vector, vector))


def normalize(vector: Sequence[Real], *, tolerance: float = 0.0) -> Vector:
    """Return a unit vector; reject vectors whose norm is at most tolerance."""
    values = as_vector(vector)
    threshold = finite_scalar(tolerance, name="tolerance")
    if threshold < 0:
        raise ValueError("tolerance must be non-negative")
    norm = magnitude(values)
    if norm <= threshold:
        raise ValueError("cannot normalize a zero vector")
    return [value / norm for value in values]
