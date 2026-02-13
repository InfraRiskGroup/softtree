# %%
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

#%%
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n_samples = 10000
    n_features, n_classes = 2, 4

    # Generate data
    Z_raw, y = make_gaussian_quantiles(
        mean=None,         # Center at (0,0)
        cov=1.0,           # Covariance
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=rng,
    )

    # Manual add noise to create overlap
    noise_level = 0.0
    # noise = rng.normal(loc=0.0, scale=noise_level, size=Z_raw.shape)
    noise = 0.0
    Z = Z_raw + noise

    # Create mask if gaps are desired
    gap = 0.0
    radii = np.linalg.norm(Z, axis=1)
    boundaries = np.percentile(radii, np.linspace(0, 100, n_classes + 1)[1:-1])
    keep_mask = np.ones(len(Z), dtype=bool)
    for b in boundaries:
        # Mark points as False if they are within 'gap' distance of a boundary
        is_in_gap = (radii > b - gap) & (radii < b + gap)
        keep_mask[is_in_gap] = False
    Z = Z[keep_mask]
    y = y[keep_mask]
    print(f"Number of samples: {len(Z)}")

    # Apply transformation
    corr_strength = 2.0
    T_mtx = np.array([
        [corr_strength, 1.0],
        [corr_strength, -1.0],
    ])
    X = Z @ T_mtx.T

    # train test split
    X_holdout, X_test, y_holdout, y_test = train_test_split(
        X, y, test_size=int(0.2*n_samples), random_state=rng
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_holdout, y_holdout, test_size=int(0.2*n_samples), random_state=24
    )

    # save data
    np.savez(
        f"data/make_gaussian_{n_samples}_seed42.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        T_mtx=T_mtx,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        noise_level=noise_level,
        gap=gap,
        corr_strength=corr_strength,
    )

    # Visualize data
    with sns.plotting_context("notebook", font_scale=1.0):
        fig, ax = plt.subplots(tight_layout=True)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=25)
        ax.set_title("make_gaussian_quantiles (4 Classes)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        plt.show()
# %%
# pytorch debugging

def _test_pytorch_results():
    X = np.array([[-2.6350, -3.8968]])

    W = np.array([
        [-0.3971, -0.3641],
        [ 0.3989,  0.4244],
        [ 0.1910,  0.0530]
    ])
    b = np.array([-0.4567, 0.0619, -0.0895])

    z = X @ W.T + b
    print(z)

    leaf_total_logs = np.array([
        [-6.3272, -5.9819, -4.8356, -3.0744],
        [-8.4199, -7.5543, -6.7361, -5.8706],
        [-3.7420, -0.8810, -3.3467, -2.0715],
        [-1.2289, -4.1693, -4.1349, -4.3106]
    ])

    logsumexp_test = np.log(np.exp(leaf_total_logs).sum(axis=0))
    print(logsumexp_test)
# %%
