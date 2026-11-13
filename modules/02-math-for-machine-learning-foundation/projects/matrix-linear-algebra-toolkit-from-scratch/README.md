# Matrix & Linear Algebra Toolkit from Scratch

An educational Python library implementing core vector, matrix, linear-system, eigenvalue, and PCA operations with Python lists and loops. NumPy is used only by the test suite as an independent numerical reference; it is not a runtime dependency and does not implement toolkit algorithms.

## Features

- Vector addition, subtraction, scalar multiplication, dot product, L2 magnitude, and normalization.
- Matrix addition, subtraction, scalar multiplication, matrix multiplication, transpose, identity, and matrix-vector multiplication.
- Partial-pivoting row echelon form and reduced row echelon form, numerical rank, singularity detection, and linear system solving.
- General solutions for underdetermined systems, including a particular solution and nullspace basis; inconsistent systems raise a dedicated exception.
- Dominant eigenpair approximation by power iteration and a symmetric Jacobi eigendecomposition used by PCA.
- PCA fitting, sample covariance, explained variance ratios, transform, and inverse transform.
- A Matplotlib demo showing PCA projection and reconstruction.

## Mathematical foundations

For vectors u,v in R^n:

- Addition: (u + v)_i = u_i + v_i; scalar multiplication: (alpha u)_i = alpha u_i.
- Dot product: u · v = sum_i(u_i v_i); Euclidean magnitude: ||u||_2 = sqrt(u · u).
- Normalization: u_hat = u / ||u||_2, defined only for nonzero vectors.

For compatible matrices, (AB)_ij = sum_k(A_ik B_kj), and transpose swaps row and column indices. Gaussian elimination applies elementary row operations to solve Ax = b. Pivoting chooses the largest-magnitude available entry in each pivot column. The solver returns one particular x_p and basis vectors n_j for the nullspace, so every solution is x = x_p + sum_j(c_j n_j).

Power iteration repeatedly applies v_(k+1) = A v_k / ||A v_k||_2, then estimates the eigenvalue with the Rayleigh quotient lambda = v^T A v. The residual ||Av - lambda v||_2 determines convergence. PCA centers sample rows, computes C = X_c^T X_c / (m - 1), and selects the top eigenvectors of the symmetric covariance matrix. The explained variance ratio is lambda_i / sum_j(lambda_j).

## Architecture

```text
src/linear_algebra_toolkit/
├── _validation.py   shared finite-number, vector, and matrix validation
├── vector.py        vector operations
├── matrix.py        matrix operations
├── gaussian.py      elimination, rank, and linear-system solution
├── eigen.py         power iteration and symmetric Jacobi eigendecomposition
└── pca.py           PCA model and projection methods
tests/               behavior, edge-case, and NumPy reference tests
examples/demo.py     end-to-end examples and PCA visualization
```

## Installation

Requires Python 3.10 or newer. The library has no runtime dependencies.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -e ".[test,demo]"
```

Install only `.[test]` for pytest and NumPy verification, or `.[demo]` for the Matplotlib demo.

## Usage

```python
from linear_algebra_toolkit import PCA, power_iteration, solve_square
from linear_algebra_toolkit import matrix, vector

print(vector.dot([1, 2, 3], [4, 5, 6]))
print(matrix.multiply([[1, 2], [3, 4]], [[2, 0], [1, 2]]))
print(solve_square([[2, 1], [1, -1]], [5, 1]))
print(power_iteration([[4, 1], [1, 3]]))

data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2]]
pca = PCA(n_components=1)
scores = pca.fit_transform(data)
print(pca.explained_variance_ratio_)
print(pca.inverse_transform(scores))
```

For underdetermined systems, `solve(A, b)` returns a `LinearSystemSolution` with `.particular`, `.nullspace_basis`, `.rank`, `.pivot_columns`, and `.is_unique`. `InconsistentSystemError` signals no solution; `solve_square` raises `SingularMatrixError` when a unique square-system solution does not exist.

## Demo

```bash
python examples/demo.py
```

The demo prints vector, matrix, solver, and eigenpair results and saves `outputs/pca_demo.png`. Add `--show` to open the chart interactively or `--output path/to/chart.png` to choose another destination.

## Tests and independent NumPy verification

```bash
python -m pytest
```

The tests cover normal operations, invalid and empty inputs, dimension mismatches, zero vectors, rank deficiency, inconsistent and underdetermined systems, solver precision, eigenpair residuals and convergence limits, PCA shape and variance behavior, and reference comparisons. `tests/test_numpy_verification.py` compares toolkit results with NumPy vector/matrix operations, `numpy.linalg.solve`, symmetric eigenvalues, and covariance/PCA results. These are independent checks only; the `src/` package imports no NumPy.

## Limitations and future improvements

- The library uses Python floats and absolute pivot tolerances; it is intended for learning and small examples, not high-performance or ill-conditioned production workloads.
- Power iteration expects a real dominant eigenvalue separated in magnitude from the rest. It does not compute complex eigenpairs; it raises `ConvergenceError` when the iteration cap is reached.
- The Jacobi eigensolver accepts real symmetric matrices only.
- PCA expects observations in rows and uses sample covariance (n - 1). It does not impute missing data or standardize feature scales automatically.
- Future work could add improved scaling-aware tolerances, QR/SVD methods, incremental PCA, and broader property-based tests while retaining the from-scratch constraint.
