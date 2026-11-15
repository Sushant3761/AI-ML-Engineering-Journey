"""Gaussian elimination, rank checks, and linear system solving."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from ._validation import Matrix, Vector, as_matrix, as_vector, finite_scalar


class InconsistentSystemError(ValueError):
    """Raised when a linear system has no solution."""


class SingularMatrixError(ValueError):
    """Raised when a square system does not have a unique solution."""


@dataclass(frozen=True)
class LinearSystemSolution:
    """A particular solution plus a basis for all null-space directions."""

    particular: Vector
    nullspace_basis: list[Vector]
    rank: int
    pivot_columns: tuple[int, ...]

    @property
    def is_unique(self) -> bool:
        """Whether this solution is the only solution."""
        return not self.nullspace_basis


def _tolerance(tolerance: Real) -> float:
    value = finite_scalar(tolerance, name="tolerance")
    if value <= 0:
        raise ValueError("tolerance must be positive")
    return value


def _eliminate(values: Matrix, tolerance: float, *, reduced: bool) -> tuple[Matrix, tuple[int, ...]]:
    rows, columns = len(values), len(values[0])
    result = [row[:] for row in values]
    pivot_row = 0
    pivots: list[int] = []
    for column in range(columns):
        candidate = max(range(pivot_row, rows), key=lambda r: abs(result[r][column]), default=pivot_row)
        if pivot_row >= rows or abs(result[candidate][column]) <= tolerance:
            continue
        result[pivot_row], result[candidate] = result[candidate], result[pivot_row]
        pivot = result[pivot_row][column]
        if reduced:
            result[pivot_row] = [value / pivot for value in result[pivot_row]]
        for row in range(rows):
            if row == pivot_row or (not reduced and row < pivot_row):
                continue
            factor = result[row][column] / (1.0 if reduced else pivot)
            if abs(factor) > tolerance:
                result[row] = [value - factor * pivot_value for value, pivot_value in zip(result[row], result[pivot_row])]
        if not reduced:
            for row in range(pivot_row + 1, rows):
                if abs(result[row][column]) <= tolerance:
                    result[row][column] = 0.0
        else:
            for row in range(rows):
                if abs(result[row][column]) <= tolerance:
                    result[row][column] = 0.0
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    cleaned = [[0.0 if abs(value) <= tolerance else value for value in row] for row in result]
    return cleaned, tuple(pivots)


def row_echelon_form(matrix: Sequence[Sequence[Real]], *, tolerance: Real = 1e-12) -> Matrix:
    """Return row echelon form using partial pivoting."""
    values = as_matrix(matrix)
    result, _ = _eliminate(values, _tolerance(tolerance), reduced=False)
    return result


def reduced_row_echelon_form(matrix: Sequence[Sequence[Real]], *, tolerance: Real = 1e-12) -> Matrix:
    """Return reduced row echelon form using partial pivoting."""
    values = as_matrix(matrix)
    result, _ = _eliminate(values, _tolerance(tolerance), reduced=True)
    return result


def matrix_rank(matrix: Sequence[Sequence[Real]], *, tolerance: Real = 1e-12) -> int:
    """Return the numerical rank using reduced row echelon form."""
    values, tol = as_matrix(matrix), _tolerance(tolerance)
    _, pivots = _eliminate(values, tol, reduced=True)
    return len(pivots)


def is_singular(matrix: Sequence[Sequence[Real]], *, tolerance: Real = 1e-12) -> bool:
    """Return whether a square matrix is singular at the given tolerance."""
    values = as_matrix(matrix)
    if len(values) != len(values[0]):
        raise ValueError("singularity is defined here only for square matrices")
    return matrix_rank(values, tolerance=tolerance) < len(values)


def solve(matrix: Sequence[Sequence[Real]], right_hand_side: Sequence[Real], *, tolerance: Real = 1e-12) -> LinearSystemSolution:
    """Solve ``A x = b``, including underdetermined systems.

    Returns one particular solution and a basis for the homogeneous nullspace.
    Raises :class:`InconsistentSystemError` when no solution exists.
    """
    coefficients = as_matrix(matrix, name="matrix")
    values = as_vector(right_hand_side, name="right_hand_side")
    if len(coefficients) != len(values):
        raise ValueError(f"matrix row count ({len(coefficients)}) must match right_hand_side length ({len(values)})")
    tol = _tolerance(tolerance)
    augmented = [row[:] + [target] for row, target in zip(coefficients, values)]
    rref, pivots = _eliminate(augmented, tol, reduced=True)
    variable_count = len(coefficients[0])
    for row in rref:
        if all(abs(value) <= tol for value in row[:variable_count]) and abs(row[-1]) > tol:
            raise InconsistentSystemError("linear system is inconsistent")
    variable_pivots = tuple(column for column in pivots if column < variable_count)
    pivot_row_by_column = {column: row for row, column in enumerate(variable_pivots)}
    particular = [0.0] * variable_count
    for column, row in pivot_row_by_column.items():
        particular[column] = rref[row][-1]
    free_columns = [column for column in range(variable_count) if column not in pivot_row_by_column]
    basis: list[Vector] = []
    for free_column in free_columns:
        direction = [0.0] * variable_count
        direction[free_column] = 1.0
        for pivot_column, row in pivot_row_by_column.items():
            direction[pivot_column] = -rref[row][free_column]
        basis.append(direction)
    return LinearSystemSolution(particular, basis, len(variable_pivots), variable_pivots)


def solve_square(matrix: Sequence[Sequence[Real]], right_hand_side: Sequence[Real], *, tolerance: Real = 1e-12) -> Vector:
    """Solve a square system that has a unique solution."""
    values = as_matrix(matrix, name="matrix")
    if len(values) != len(values[0]):
        raise ValueError("solve_square requires a square coefficient matrix")
    result = solve(values, right_hand_side, tolerance=tolerance)
    if not result.is_unique:
        raise SingularMatrixError("coefficient matrix is singular; solution is not unique")
    return result.particular
