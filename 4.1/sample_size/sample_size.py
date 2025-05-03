import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm

def rbf_kernel(X, Y, sigma):
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

def compute_mmd2(X, Y, sigma):
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    n, m = len(X), len(Y)
    return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()

def permutation_test(X, Y, sigma, num_perm=100):
    Z = np.vstack([X, Y])
    n = len(X)
    real_mmd = compute_mmd2(X, Y, sigma)
    mmd_null = [compute_mmd2(np.random.permutation(Z)[:n], np.random.permutation(Z)[n:], sigma)
                for _ in range(num_perm)]
    return np.mean(np.array(mmd_null) >= real_mmd)

def median_heuristic(X, Y):
    Z = np.vstack([X, Y])
    dists = cdist(Z, Z, 'euclidean')
    return np.median(dists)

sample_sizes = list(range(10, 500, 10))
num_trials = 100
scenarios = ['1D-small', '1D-large', 'HD-small', 'HD-large',
             'Cov-small', 'Cov-large', 'Mix-small', 'Mix-large']
results = []

for scenario in tqdm(scenarios, desc="Running all scenarios"):
    for n in sample_sizes:
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

            sigma = median_heuristic(X, Y)
            p_value = permutation_test(X, Y, sigma, num_perm=100)
            if p_value < 0.05:
                power += 1

        results.append({
            'scenario': scenario,
            'sample_size': n,
            'power': power / num_trials
        })

df_results = pd.DataFrame(results)
df_results.to_csv("mmd_power_sample_size.csv", index=False)
print("Results saved to mmd_power_sample_size.csv")
