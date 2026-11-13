import pytest

from linear_algebra_toolkit import matrix


def test_matrix_arithmetic_and_shapes():
    a, b = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    assert matrix.add(a, b) == [[6.0, 8.0], [10.0, 12.0]]
    assert matrix.subtract(b, a) == [[4.0, 4.0], [4.0, 4.0]]
    assert matrix.scale(a, 2) == [[2.0, 4.0], [6.0, 8.0]]
    assert matrix.shape(a) == (2, 2)


def test_matrix_products_and_identity():
    assert matrix.multiply([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [
        [58.0, 64.0], [139.0, 154.0]
    ]
    assert matrix.multiply_vector([[1, 2], [3, 4]], [5, 6]) == [17.0, 39.0]
    assert matrix.transpose([[1, 2, 3], [4, 5, 6]]) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert matrix.multiply([[2, 3], [4, 5]], matrix.identity(2)) == [[2.0, 3.0], [4.0, 5.0]]


@pytest.mark.parametrize("bad", [[], [[]], [[1, 2], [3]], [[1], []]])
def test_matrix_empty_and_ragged_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        matrix.transpose(bad)


def test_matrix_shape_errors():
    with pytest.raises(ValueError, match="shapes differ"):
        matrix.add([[1, 2]], [[1], [2]])
    with pytest.raises(ValueError, match="incompatible"):
        matrix.multiply([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="match vector length"):
        matrix.multiply_vector([[1, 2]], [3])
    with pytest.raises(ValueError, match="positive"):
        matrix.identity(0)
