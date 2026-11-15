import pytest

from linear_algebra_toolkit import (
    InconsistentSystemError,
    SingularMatrixError,
    is_singular,
    matrix_rank,
    reduced_row_echelon_form,
    row_echelon_form,
    solve,
    solve_square,
)


def test_echelon_forms_and_rank():
    values = [[1, 2, 1], [2, 4, 0], [3, 6, 3]]
    assert row_echelon_form(values)[1][1] == 0.0
    assert reduced_row_echelon_form([[1, 2], [3, 4]]) == [[1.0, 0.0], [0.0, 1.0]]
    assert matrix_rank(values) == 2


def test_unique_solution_and_singular_detection():
    assert solve_square([[2, 1], [1, -1]], [5, 1]) == [2.0, 1.0]
    assert is_singular([[1, 2], [2, 4]])
    with pytest.raises(SingularMatrixError):
        solve_square([[1, 2], [2, 4]], [3, 6])


def test_inconsistent_system_is_reported():
    with pytest.raises(InconsistentSystemError, match="inconsistent"):
        solve([[1, 1], [1, 1]], [2, 3])


def test_underdetermined_system_returns_particular_solution_and_nullspace():
    result = solve([[1, 1]], [2])
    assert not result.is_unique
    assert result.particular == [2.0, 0.0]
    assert result.nullspace_basis == [[-1.0, 1.0]]
    assert result.rank == 1


def test_gaussian_dimension_and_empty_errors():
    with pytest.raises(ValueError, match="must not be empty"):
        matrix_rank([])
    with pytest.raises(ValueError, match="row count"):
        solve([[1, 2]], [1, 2])
