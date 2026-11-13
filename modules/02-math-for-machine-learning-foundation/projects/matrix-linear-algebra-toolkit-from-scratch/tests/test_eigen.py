import pytest

from linear_algebra_toolkit import ConvergenceError, power_iteration, symmetric_eigendecomposition
from linear_algebra_toolkit.matrix import multiply_vector


def test_power_iteration_converges_and_has_small_residual():
    values = [[4, 1], [1, 3]]
    result = power_iteration(values, tolerance=1e-11)
    image = multiply_vector(values, result.eigenvector)
    assert result.converged
    assert result.eigenvalue == pytest.approx(4.61803398875, rel=1e-9)
    assert sum((x - result.eigenvalue * y) ** 2 for x, y in zip(image, result.eigenvector)) ** 0.5 < 1e-10


def test_power_iteration_zero_matrix_and_input_validation():
    result = power_iteration([[0, 0], [0, 0]])
    assert result.eigenvalue == 0
    assert result.converged
    with pytest.raises(ValueError, match="square"):
        power_iteration([[1, 2]])
    with pytest.raises(ValueError, match="length"):
        power_iteration([[1, 0], [0, 1]], [1])
    with pytest.raises(ValueError, match="zero vector"):
        power_iteration([[1, 0], [0, 1]], [0, 0])


def test_power_iteration_iteration_cap_is_explicit():
    with pytest.raises(ConvergenceError, match="within 1 iterations"):
        power_iteration([[2, 0], [0, 1]], [1, 1], max_iterations=1, tolerance=1e-15)


def test_symmetric_jacobi_eigendecomposition():
    eigenvalues, eigenvectors = symmetric_eigendecomposition([[2, 1], [1, 2]])
    assert eigenvalues == pytest.approx([3, 1])
    # Eigenvectors are columns; each must satisfy A v = lambda v.
    for column, eigenvalue in enumerate(eigenvalues):
        vector = [eigenvectors[row][column] for row in range(2)]
        image = multiply_vector([[2, 1], [1, 2]], vector)
        assert image == pytest.approx([eigenvalue * value for value in vector])
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_eigendecomposition([[1, 2], [0, 1]])
