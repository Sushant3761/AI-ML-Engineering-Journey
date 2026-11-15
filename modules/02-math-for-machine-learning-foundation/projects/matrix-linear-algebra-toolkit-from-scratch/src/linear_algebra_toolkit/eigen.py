"""Eigenvalue routines written without numerical-array dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from numbers import Real
from typing import Sequence

from ._validation import Matrix, Vector, as_matrix, as_vector, finite_scalar
from .matrix import multiply_vector
from .vector import dot, magnitude, normalize


class ConvergenceError(RuntimeError):
    """Raised when an iterative eigenvalue algorithm reaches its iteration cap."""


@dataclass(frozen=True)
class Eigenpair:
    """Approximate real eigenvalue and normalized eigenvector."""

    eigenvalue: float
    eigenvector: Vector
    iterations: int
    converged: bool


def _square_matrix(matrix: Sequence[Sequence[Real]]) -> Matrix:
    values = as_matrix(matrix)
    if len(values) != len(values[0]):
        raise ValueError("eigenvalue routines require a square matrix")
    return values


def power_iteration(
    matrix: Sequence[Sequence[Real]],
    initial_vector: Sequence[Real] | None = None,
    *,
    tolerance: Real = 1e-10,
    max_iterations: int = 1000,
) -> Eigenpair:
    """Estimate the dominant-magnitude eigenpair by power iteration.

    Convergence is checked with the residual ``||A v - lambda v||``. The
    method assumes a real dominant eigenvalue separated in magnitude from
    the remaining eigenvalues.
    """
    values = _square_matrix(matrix)
    tol = finite_scalar(tolerance, name="tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    vector = normalize(initial_vector if initial_vector is not None else [1.0] * len(values))
    if len(vector) != len(values):
        raise ValueError("initial_vector length must match matrix dimension")
    for iteration in range(1, max_iterations + 1):
        product = multiply_vector(values, vector)
        product_norm = magnitude(product)
        if product_norm == 0.0:
            return Eigenpair(0.0, vector, iteration, True)
        vector = [entry / product_norm for entry in product]
        image = multiply_vector(values, vector)
        eigenvalue = dot(vector, image)
        residual = sqrt(sum((x - eigenvalue * y) ** 2 for x, y in zip(image, vector)))
        if residual <= tol:
            return Eigenpair(eigenvalue, vector, iteration, True)
    raise ConvergenceError(f"power iteration did not converge within {max_iterations} iterations")


def symmetric_eigendecomposition(
    matrix: Sequence[Sequence[Real]], *, tolerance: Real = 1e-12, max_iterations: int = 10_000
) -> tuple[list[float], Matrix]:
    """Return all eigenvalues and orthonormal eigenvectors of a symmetric matrix.

    Uses Jacobi rotations. Eigenvectors are returned as columns of the second
    result, with eigenvalues sorted from largest to smallest.
    """
    values = _square_matrix(matrix)
    tol = finite_scalar(tolerance, name="tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(values[i][j] - values[j][i]) > tol:
                raise ValueError("Jacobi eigendecomposition requires a symmetric matrix")
    a = [[0.5 * (values[i][j] + values[j][i]) for j in range(n)] for i in range(n)]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_iterations):
        p, q = max(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda pair: abs(a[pair[0]][pair[1]]), default=(0, 0))
        if p == q or abs(a[p][q]) <= tol:
            break
        angle = 0.5 * atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        cosine, sine = cos(angle), sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        a[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = cosine * akp - sine * akq
                a[k][q] = a[q][k] = sine * akp + cosine * akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = cosine * vkp - sine * vkq
            vectors[k][q] = sine * vkp + cosine * vkq
    else:
        raise ConvergenceError(f"Jacobi eigendecomposition did not converge within {max_iterations} rotations")
    order = sorted(range(n), key=lambda column: a[column][column], reverse=True)
    eigenvalues = [a[index][index] for index in order]
    eigenvectors = [[vectors[row][column] for column in order] for row in range(n)]
    return eigenvalues, eigenvectors
