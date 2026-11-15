"""Principal component analysis using pure-Python centering and eigensolving."""

from __future__ import annotations

from numbers import Real
from typing import Sequence

from ._validation import Matrix, as_matrix, finite_scalar
from .eigen import symmetric_eigendecomposition


class PCA:
    """Principal component analysis for observations stored in rows.

    Covariance uses the unbiased sample estimate with denominator ``n - 1``.
    Core calculations use Python lists and loops; no NumPy is imported here.
    """

    def __init__(self, n_components: int, *, tolerance: Real = 1e-12) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise TypeError("n_components must be an integer")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.tolerance = finite_scalar(tolerance, name="tolerance")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.mean_: list[float] | None = None
        self.covariance_: Matrix | None = None
        self.components_: Matrix | None = None  # component vectors as rows
        self.explained_variance_: list[float] | None = None
        self.explained_variance_ratio_: list[float] | None = None

    def fit(self, data: Sequence[Sequence[Real]]) -> "PCA":
        """Fit component directions from a samples-by-features dataset."""
        values = as_matrix(data, name="data")
        sample_count, feature_count = len(values), len(values[0])
        if sample_count < 2:
            raise ValueError("PCA requires at least two observations")
        if self.n_components > feature_count:
            raise ValueError("n_components cannot exceed the number of features")
        means = [sum(row[column] for row in values) / sample_count for column in range(feature_count)]
        centered = [[row[column] - means[column] for column in range(feature_count)] for row in values]
        covariance = [
            [sum(row[i] * row[j] for row in centered) / (sample_count - 1) for j in range(feature_count)]
            for i in range(feature_count)
        ]
        eigenvalues, eigenvectors = symmetric_eigendecomposition(covariance, tolerance=self.tolerance)
        total_variance = sum(max(0.0, value) for value in eigenvalues)
        selected_values = [max(0.0, value) for value in eigenvalues[: self.n_components]]
        components = [[eigenvectors[row][column] for row in range(feature_count)] for column in range(self.n_components)]
        self.mean_ = means
        self.covariance_ = covariance
        self.components_ = components
        self.explained_variance_ = selected_values
        self.explained_variance_ratio_ = (
            [value / total_variance for value in selected_values]
            if total_variance > self.tolerance
            else [0.0] * self.n_components
        )
        return self

    def transform(self, data: Sequence[Sequence[Real]]) -> Matrix:
        """Center observations and project them into the fitted components."""
        mean, components = self._fitted()
        values = as_matrix(data, name="data")
        if len(values[0]) != len(mean):
            raise ValueError("data feature count does not match fitted PCA")
        return [
            [sum((row[j] - mean[j]) * component[j] for j in range(len(mean))) for component in components]
            for row in values
        ]

    def inverse_transform(self, transformed: Sequence[Sequence[Real]]) -> Matrix:
        """Project component scores back into the original feature space."""
        mean, components = self._fitted()
        scores = as_matrix(transformed, name="transformed")
        if len(scores[0]) != len(components):
            raise ValueError("transformed feature count does not match fitted component count")
        return [
            [mean[j] + sum(score[k] * components[k][j] for k in range(len(components))) for j in range(len(mean))]
            for score in scores
        ]

    def fit_transform(self, data: Sequence[Sequence[Real]]) -> Matrix:
        """Fit PCA and return the transformed training observations."""
        return self.fit(data).transform(data)

    def _fitted(self) -> tuple[list[float], Matrix]:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform or inverse_transform")
        return self.mean_, self.components_
