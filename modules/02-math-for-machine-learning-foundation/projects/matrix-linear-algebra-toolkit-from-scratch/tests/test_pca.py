import pytest

from linear_algebra_toolkit import PCA


def test_pca_fit_transform_inverse_and_variance():
    data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7]]
    model = PCA(1)
    scores = model.fit_transform(data)
    reconstructed = model.inverse_transform(scores)
    assert len(scores) == len(data)
    assert all(len(row) == 1 for row in scores)
    assert all(len(row) == 2 for row in reconstructed)
    assert 0.9 < model.explained_variance_ratio_[0] < 1.0
    assert model.explained_variance_ratio_[0] > 0.9
    assert model.mean_ == pytest.approx([sum(row[j] for row in data) / len(data) for j in range(2)])


def test_pca_full_components_reconstructs_data():
    data = [[1, 2], [2, 0], [4, 5], [3, 2]]
    model = PCA(2)
    reconstructed = model.fit(data).inverse_transform(model.transform(data))
    for actual, estimate in zip(data, reconstructed):
        assert estimate == pytest.approx(actual, abs=1e-10)


def test_pca_dimensions_empty_inputs_and_fit_requirement():
    with pytest.raises(RuntimeError, match="fitted"):
        PCA(1).transform([[1, 2]])
    with pytest.raises(ValueError, match="at least two"):
        PCA(1).fit([[1, 2]])
    with pytest.raises(ValueError, match="n_components"):
        PCA(3).fit([[1, 2], [2, 3]])
    with pytest.raises(ValueError, match="not be empty"):
        PCA(1).fit([])
