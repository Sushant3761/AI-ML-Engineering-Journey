import math

import pytest

from linear_algebra_toolkit import vector


def test_vector_arithmetic_and_norms():
    assert vector.add([1, 2], [3, 4]) == [4.0, 6.0]
    assert vector.subtract([3, 4], [1, 2]) == [2.0, 2.0]
    assert vector.scale([1, -2], 3) == [3.0, -6.0]
    assert vector.dot([1, 2, 3], [4, 5, 6]) == 32
    assert vector.magnitude([3, 4]) == 5
    assert vector.normalize([3, 4]) == [0.6, 0.8]


@pytest.mark.parametrize("operation", [vector.add, vector.subtract, vector.dot])
def test_vector_dimensions_must_match(operation):
    with pytest.raises(ValueError, match="dimensions differ"):
        operation([1], [1, 2])


@pytest.mark.parametrize("values", [[], [0, 0]])
def test_vector_empty_and_zero_norm_errors(values):
    if not values:
        with pytest.raises(ValueError, match="must not be empty"):
            vector.magnitude(values)
    else:
        with pytest.raises(ValueError, match="zero vector"):
            vector.normalize(values)


def test_vector_rejects_non_finite_and_non_numeric_inputs():
    with pytest.raises(ValueError, match="finite"):
        vector.add([math.inf], [1])
    with pytest.raises(TypeError, match="real numbers"):
        vector.dot(["1"], [1])
