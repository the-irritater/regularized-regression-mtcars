"""
Coefficient Shrinkage Path Trace Plot Generator
================================================
Generates regularization path trace plots for Ridge and Lasso regression
on the mtcars dataset, demonstrating parameter shrinkage and feature selection.

Usage:
    python trace_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler


def load_mtcars_data() -> pd.DataFrame:
    """Load or generate mtcars dataset."""
    url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        np.random.seed(42)
        n = 32
        df = pd.DataFrame({
            "mpg": np.random.uniform(10, 35, n),
            "cyl": np.random.choice([4, 6, 8], n),
            "disp": np.random.uniform(70, 470, n),
            "hp": np.random.uniform(50, 335, n),
            "drat": np.random.uniform(2.7, 4.9, n),
            "wt": np.random.uniform(1.5, 5.4, n),
            "qsec": np.random.uniform(14, 23, n),
        })
    return df


def plot_shrinkage_paths():
    print("=" * 60)
    print("GENERATING RIDGE & LASSO SHRINKAGE TRACE PLOTS")
    print("=" * 60)

    df = load_mtcars_data()
    # Select only numeric feature columns
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != "mpg"]
    
    X = numeric_df[feature_cols]
    y = numeric_df["mpg"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = np.logspace(-3, 3, 200)

    ridge_coefs = []
    lasso_coefs = []

    for a in alphas:
        ridge = Ridge(alpha=a)
        ridge.fit(X_scaled, y)
        ridge_coefs.append(ridge.coef_)

        lasso = Lasso(alpha=a, max_iter=10000)
        lasso.fit(X_scaled, y)
        lasso_coefs.append(lasso.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    plt.figure(figsize=(14, 6))

    # Subplot 1: Ridge Trace
    plt.subplot(1, 2, 1)
    for i, col in enumerate(feature_cols):
        plt.plot(alphas, ridge_coefs[:, i], label=col, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Alpha (L2 Penalty)")
    plt.ylabel("Standardized Coefficients")
    plt.title("Ridge Regression Coefficient Paths")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Lasso Trace
    plt.subplot(1, 2, 2)
    for i, col in enumerate(feature_cols):
        plt.plot(alphas, lasso_coefs[:, i], label=col, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Alpha (L1 Penalty)")
    plt.ylabel("Standardized Coefficients")
    plt.title("Lasso Regression Coefficient Paths (Feature Selection)")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "Images")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "coefficient_shrinkage_paths.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved coefficient shrinkage trace plot to {out_file}")


if __name__ == "__main__":
    plot_shrinkage_paths()
