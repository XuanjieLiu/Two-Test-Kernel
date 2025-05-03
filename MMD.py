import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def rbf_kernel(X, Y, sigma=1.0):
    """Compute the RBF (Gaussian) kernel matrix between X and Y"""
    dists = cdist(X, Y, 'sqeuclidean')  # shape (n, m)
    return np.exp(-dists / (2 * sigma ** 2))

def compute_mmd2(X, Y, sigma=1.0):
    """Compute unbiased estimate of MMD^2 between X and Y"""
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    n = X.shape[0]
    m = Y.shape[0]

    # remove diagonals (unbiased estimator)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd2 = K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2 * K_XY.mean()
    return mmd2

def permutation_test(X, Y, num_permutations=500, sigma=1.0):
    """Run permutation test for MMD^2"""
    Z = np.vstack([X, Y])
    n = X.shape[0]
    mmd_obs = compute_mmd2(X, Y, sigma)
    mmd_null = []

    for _ in range(num_permutations):
        perm = np.random.permutation(Z)
        X_perm = perm[:n]
        Y_perm = perm[n:]
        mmd_null.append(compute_mmd2(X_perm, Y_perm, sigma))

    p_value = np.mean(np.array(mmd_null) >= mmd_obs)

    return mmd_obs, p_value, mmd_null

# ==== Example use ====
def demo():
    # Generate toy dataset
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 2))  # N(0, I)
    Y = np.random.normal(0, 1, size=(100, 2))  # N(0, I), same distribution

    # Run test
    sigma = 1.0
    mmd_stat, p_val, null_dist = permutation_test(X, Y, sigma=sigma)

    print(f"MMD^2: {mmd_stat:.4f}")
    print(f"p-value: {p_val:.4f}")

    # Optional: visualize null distribution
    plt.hist(null_dist, bins=30, alpha=0.7, label="Null MMD^2")
    plt.axvline(mmd_stat, color='red', linestyle='--', label="Observed MMD^2")
    plt.title("Permutation Test for MMD^2")
    plt.xlabel("MMD^2")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    demo()