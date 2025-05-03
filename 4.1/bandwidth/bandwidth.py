import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Define RBF kernel with arbitrary sigma
def rbf_kernel_sigma(X, Y, sigma):
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

# MMD^2 computation
def compute_mmd2_rbf(X, Y, sigma):
    Kxx = rbf_kernel_sigma(X, X, sigma)
    Kyy = rbf_kernel_sigma(Y, Y, sigma)
    Kxy = rbf_kernel_sigma(X, Y, sigma)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    n, m = len(X), len(Y)
    return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()

# Permutation test
def permutation_test_rbf(X, Y, sigma, num_perm=20):
    Z = np.vstack([X, Y])
    n = len(X)
    real_mmd = compute_mmd2_rbf(X, Y, sigma)
    mmd_null = []
    for _ in range(num_perm):
        idx = np.random.permutation(len(Z))
        X_perm = Z[idx[:n]]
        Y_perm = Z[idx[n:]]
        mmd_null.append(compute_mmd2_rbf(X_perm, Y_perm, sigma))
    return np.mean(np.array(mmd_null) >= real_mmd)

# Median heuristic
def median_heuristic(X, Y):
    Z = np.vstack([X, Y])
    dists = cdist(Z, Z, 'euclidean')
    return np.median(dists)

# Parameters
n = 100
num_trials = 100
scenarios = ['1D-small', '1D-large', 'HD-small', 'HD-large',
             'Cov-small', 'Cov-large', 'Mix-small', 'Mix-large']
bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
results_bandwidth = []

# Run experiments for fixed bandwidths
for sigma in tqdm(bandwidths, desc="Testing fixed RBF bandwidths"):
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
                X = np.concatenate([np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
                Y = np.concatenate([np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
            elif scenario == 'Mix-large':
                means_P = [-1, 1]
                means_Q = [-3, 3]
                X = np.concatenate([np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
                Y = np.concatenate([np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
            else:
                continue

            p_value = permutation_test_rbf(X, Y, sigma, num_perm=20)
            if p_value < 0.05:
                power += 1

        results_bandwidth.append({
            'kernel': 'rbf',
            'sigma': sigma,
            'scenario': scenario,
            'sample_size': n,
            'power': power / num_trials,
            'method': 'fixed'
        })

# Add median heuristic as a separate method
for scenario in tqdm(scenarios, desc="Testing median heuristic"):
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
            X = np.concatenate([np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
            Y = np.concatenate([np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
        elif scenario == 'Mix-large':
            means_P = [-1, 1]
            means_Q = [-3, 3]
            X = np.concatenate([np.random.normal(loc=means_P[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
            Y = np.concatenate([np.random.normal(loc=means_Q[i % 2], scale=1.0, size=(n // 2, 1)) for i in range(2)])
        else:
            continue

        sigma = median_heuristic(X, Y)
        p_value = permutation_test_rbf(X, Y, sigma, num_perm=20)
        if p_value < 0.05:
            power += 1

    results_bandwidth.append({
        'kernel': 'rbf',
        'sigma': sigma,
        'scenario': scenario,
        'sample_size': n,
        'power': power / num_trials,
        'method': 'median'
    })

# Save results
df_bandwidth = pd.DataFrame(results_bandwidth)
df_bandwidth.to_csv("./mmd_power_rbf_bandwidth_full.csv", index=False)
