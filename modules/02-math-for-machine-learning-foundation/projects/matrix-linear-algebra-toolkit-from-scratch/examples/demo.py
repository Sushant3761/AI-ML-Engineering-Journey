"""Run a small toolkit tour and save a PCA visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from linear_algebra_toolkit import PCA, power_iteration, solve_square
from linear_algebra_toolkit import matrix, vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="also open the plot in a window")
    parser.add_argument("--output", type=Path, default=Path("outputs/pca_demo.png"))
    args = parser.parse_args()

    print("vector sum:", vector.add([1, 2, 3], [4, 5, 6]))
    print("matrix product:", matrix.multiply([[1, 2], [3, 4]], [[2, 0], [1, 2]]))
    print("linear system solution:", solve_square([[3, 1], [1, 2]], [9, 8]))
    print("dominant eigenpair:", power_iteration([[4, 1], [1, 3]]))

    data = [
        [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
        [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9],
    ]
    pca = PCA(n_components=1)
    scores = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(scores)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter([row[0] for row in data], [row[1] for row in data], label="observations")
    axis.scatter([row[0] for row in reconstructed], [row[1] for row in reconstructed], marker="x", label="1D PCA reconstruction")
    for original, projected in zip(data, reconstructed):
        axis.plot([original[0], projected[0]], [original[1], projected[1]], color="gray", alpha=0.35)
    axis.set(title="PCA projection and reconstruction", xlabel="Feature 1", ylabel="Feature 2")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"PCA visualization saved to {args.output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
