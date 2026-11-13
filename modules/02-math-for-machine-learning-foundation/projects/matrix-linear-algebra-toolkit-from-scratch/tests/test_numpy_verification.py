"""Independent reference checks; NumPy is never imported by the library."""

import numpy as np
import pytest

from linear_algebra_toolkit import PCA, matrix, power_iteration, solve_square, vector


def test_vectors_against_numpy():
    left, right = [0.25, -2.0, 7.5], [3.0, 4.5, -1.0]
    a, b = np.array(left), np.array(right)
    assert vector.add(left, right) == pytest.approx((a + b).tolist())
    assert vector.subtract(left, right) == pytest.approx((a - b).tolist())
    assert vector.scale(left, 1.7) == pytest.approx((a * 1.7).tolist())
    assert vector.dot(left, right) == pytest.approx(float(a @ b))
    assert vector.magnitude(left) == pytest.approx(float(np.linalg.norm(a)))


def test_matrices_against_numpy():
    left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    right = [[2.0, 0.0], [1.0, 3.0], [-1.0, 4.0]]
    a, b = np.array(left), np.array(right)
    assert np.allclose(np.array(matrix.multiply(left, right)), a @ b)
    assert np.allclose(np.array(matrix.transpose(left)), a.T)
    assert matrix.multiply_vector(left, [2, -1, 0.5]) == pytest.approx((a @ np.array([2, -1, 0.5])).tolist())


def test_linear_solve_against_numpy():
    coefficients = [[4.0, 2.0, 1.0], [1.0, 5.0, 2.0], [2.0, 1.0, 6.0]]
    target = [7.0, 8.0, 9.0]
    assert solve_square(coefficients, target) == pytest.approx(np.linalg.solve(coefficients, target).tolist())


def test_power_iteration_against_numpy_eigenvalue():
    values = [[5.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]]
    actual = power_iteration(values, tolerance=1e-11, max_iterations=5000)
    expected = max(np.linalg.eigvalsh(np.array(values)))
    assert actual.eigenvalue == pytest.approx(float(expected), rel=1e-9, abs=1e-10)


def test_pca_eigenvalues_and_subspace_against_numpy():
    data = [[1.0, 2.0, 5.0], [2.0, 1.0, 4.0], [3.0, 4.0, 6.0], [4.0, 3.0, 8.0], [5.0, 7.0, 9.0]]
    expected_values, expected_vectors = np.linalg.eigh(np.cov(np.array(data), rowvar=False))
    order = np.argsort(expected_values)[::-1]
    model = PCA(2).fit(data)
    assert model.explained_variance_ == pytest.approx(expected_values[order[:2]].tolist(), rel=1e-9)
    assert sum(model.explained_variance_ratio_) == pytest.approx(
        float(expected_values[order[:2]].sum() / expected_values.sum()), rel=1e-9
    )
    expected_scores = (np.array(data) - np.mean(data, axis=0)) @ expected_vectors[:, order[:2]]
    # Signs and rotations within repeated-eigenvalue spaces are not unique.
    assert np.abs(np.corrcoef(np.array(model.transform(data)).T, expected_scores.T)[:2, 2:]).diagonal() == pytest.approx(
        [1.0, 1.0], abs=1e-8
    )
