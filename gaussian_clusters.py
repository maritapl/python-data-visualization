import numpy as np
import matplotlib.pyplot as plt

# Scatter plot with three Gaussian clusters
# Each cluster has its own mean (center) and covariance (shape/tilt).
# We also mark and label the cluster centers.
# Equal aspect keeps x/y units consistent (so shapes aren’t distorted).

def main():
    np.random.seed(42)
    n = 300

    mu1 = np.array([0.0, 0.0])
    cov1 = np.array([[1.0, 0.2],
                     [0.2, 0.8]])

    mu2 = np.array([6.0, 2.0])
    cov2 = np.array([[1.5, -0.6],
                     [-0.6, 1.0]])

    mu3 = np.array([2.0, 7.0])
    cov3 = np.array([[0.6, 0.0],
                     [0.0, 1.8]])

    c1 = np.random.multivariate_normal(mu1, cov1, size=n)
    c2 = np.random.multivariate_normal(mu2, cov2, size=n)
    c3 = np.random.multivariate_normal(mu3, cov3, size=n)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(c1[:, 0], c1[:, 1], marker="o", alpha=0.8, label="Cluster 1")
    ax.scatter(c2[:, 0], c2[:, 1], marker=">", alpha=0.8, label="Cluster 2")
    ax.scatter(c3[:, 0], c3[:, 1], marker="^", alpha=0.8, label="Cluster 3")

    centers = np.vstack([mu1, mu2, mu3])
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=140, label="Centers")

    for i, (cx, cy) in enumerate(centers, start=1):
        ax.text(cx, cy, f"  C{i}", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter plot with 3 Gaussian clusters")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
