"""A small educational linear algebra toolkit implemented from scratch."""

from .eigen import ConvergenceError, Eigenpair, power_iteration, symmetric_eigendecomposition
from .gaussian import (
    InconsistentSystemError,
    LinearSystemSolution,
    SingularMatrixError,
    is_singular,
    matrix_rank,
    reduced_row_echelon_form,
    row_echelon_form,
    solve,
    solve_square,
)
from .pca import PCA

__all__ = [
    "ConvergenceError",
    "Eigenpair",
    "InconsistentSystemError",
    "LinearSystemSolution",
    "PCA",
    "SingularMatrixError",
    "is_singular",
    "matrix_rank",
    "power_iteration",
    "reduced_row_echelon_form",
    "row_echelon_form",
    "solve",
    "solve_square",
    "symmetric_eigendecomposition",
]
