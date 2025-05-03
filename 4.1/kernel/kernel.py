import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm

# Define 6+ different kernel functions for comparison
def rbf_kernel_fixed(X, Y, sigma=1.0):
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

def linear_kernel(X, Y):
    return X @ Y.T

def poly_kernel(X, Y, degree=3, coef0=1):
    return (X @ Y.T + coef0) ** degree

def laplace_kernel(X, Y, sigma=1.0):
    dists = cdist(X, Y, 'euclidean')
    return np.exp(-dists / sigma)

def cosine_kernel(X, Y):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X_norm @ Y_norm.T

def sigmoid_kernel(X, Y, alpha=0.01, coef0=0.0):
    return np.tanh(alpha * (X @ Y.T) + coef0)


def compute_mmd2_kernel(X, Y, kernel_func):
    Kxx = kernel_func(X, X)
    Kyy = kernel_func(Y, Y)
    Kxy = kernel_func(X, Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    n, m = len(X), len(Y)
    return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()

def permutation_test_kernel(X, Y, kernel_func, num_perm=100):
    Z = np.vstack([X, Y])
    n = len(X)
    real_mmd = compute_mmd2_kernel(X, Y, kernel_func)
    mmd_null = []
    for _ in range(num_perm):
        idx = np.random.permutation(len(Z))
        X_perm = Z[idx[:n]]
        Y_perm = Z[idx[n:]]
        mmd_null.append(compute_mmd2_kernel(X_perm, Y_perm, kernel_func))
    return np.mean(np.array(mmd_null) >= real_mmd)


# Prepare kernel list
kernel_choices = {
    'linear': linear_kernel,
    'polynomial': lambda X, Y: poly_kernel(X, Y, degree=3, coef0=1),
    'rbf': lambda X, Y: rbf_kernel_fixed(X, Y, sigma=1.0),
    'laplace': lambda X, Y: laplace_kernel(X, Y, sigma=1.0),
    'cosine': cosine_kernel,
    'sigmoid': sigmoid_kernel
}

# Parameters
n = 100
num_trials = 100
scenarios = ['1D-small', '1D-large', 'HD-small', 'HD-large',
             'Cov-small', 'Cov-large', 'Mix-small', 'Mix-large']
results_kernel = []

# Run experiment
for kernel_name, kernel_func in kernel_choices.items():
    for scenario in scenarios:
        power = 0
        for _ in range(num_trials):
            if scenario == '1D-small':
                X = np.random.normal(0, 1, (n, 1))
                Y = np.random.normal(0.2, 1, (n, 1))
            elif scenario == '1D-large':
                X = np.random.normal(0, 1, (n, 1))
                Y = np.random.normal(1.0, 1, (n, 1))
            elif scenario == 'HD-small':
                d = 10
                X = np.random.normal(0, 1, (n, d))
                Y = np.random.normal(0.2, 1, (n, d))
            elif scenario == 'HD-large':
                d = 100
                X = np.random.normal(0, 1, (n, d))
                Y = np.random.normal(1.0, 1, (n, d))
            elif scenario == 'Cov-small':
                d = 10
                X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)
                Sigma = np.diag([1.5 if i % 2 == 0 else 1.0 for i in range(d)])
                Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
            elif scenario == 'Cov-large':
                d = 10
                X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)
                Sigma = np.diag([4.0 if i % 2 == 0 else 1.0 for i in range(d)])
                Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
            elif scenario == 'Mix-small':
                means_P = [-1, 1]
                means_Q = [-1.5, 1.5]
                X = np.concatenate([
                    np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)
                ])
                Y = np.concatenate([
                    np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)
                ])
            elif scenario == 'Mix-large':
                means_P = [-1, 1]
                means_Q = [-3, 3]
                X = np.concatenate([
                    np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)
                ])
                Y = np.concatenate([
                    np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)
                ])
            else:
                continue

            p_value = permutation_test_kernel(X, Y, kernel_func, num_perm=100)
            if p_value < 0.05:
                power += 1

        results_kernel.append({
            'kernel': kernel_name,
            'scenario': scenario,
            'sample_size': n,
            'power': power / num_trials
        })
        print(f"Kernel: {kernel_name}, Scenario: {scenario}, Power: {power / num_trials:.4f}")
# Convert to DataFrame and show
df_results = pd.DataFrame(results_kernel)
df_results.to_csv("./mmd_power_kernel.csv", index=False)
