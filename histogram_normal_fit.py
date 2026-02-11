import numpy as np
import matplotlib.pyplot as plt

# Histogram with normal fit overlay
# Plot histogram as density (area=1)
# Compute sample mean (mu) and sample std (sigma)
# Overlay the Normal PDF using mu/sigma from the data

def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    np.random.seed(42)

    data = np.random.normal(loc=0.0, scale=1.0, size=1000)

    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))  # ddof=1 -> sample std

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(data, bins=30, density=True, alpha=0.7, label="Histogram (density)")

    x = np.linspace(data.min(), data.max(), 300)
    pdf = normal_pdf(x, mu, sigma)
    ax.plot(x, pdf, label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram with normal fit overlay")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
