"""
Regularized Regression Analysis on the mtcars Dataset
=====================================================

A comparative study of OLS, Ridge, and Lasso regression models for
predicting fuel efficiency (mpg) with emphasis on handling
multicollinearity through regularization techniques.

Problem Statement
-----------------
The mtcars dataset contains 10 mechanical and design attributes for 32
automobiles, many of which are highly correlated (multicollinearity).
Standard OLS regression produces unstable and unreliable coefficient
estimates under these conditions, leading to poor predictive performance
and misleading feature importance. This project investigates how
regularization techniques can build more accurate and interpretable
models for predicting fuel efficiency (mpg).

Objectives
----------
1. Build an OLS regression model as a baseline for predicting mpg.
2. Apply Ridge Regression (L2) to stabilize coefficient estimates and
   improve prediction accuracy under multicollinearity.
3. Apply Lasso Regression (L1) to perform automatic feature selection
   and identify the most influential predictors.
4. Compare all three models using R-squared and MSE on a held-out test set.
5. Analyze coefficient behavior across models to demonstrate the effect
   of regularization on multicollinear data.
6. Identify the key vehicle attributes driving fuel efficiency.

Author : Sanman Kadam
Email  : sanman.kadam@statistics.mu.ac.in
Date   : April 2026
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Plot configuration
sns.set_context("notebook")
sns.set_style("white")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# 1. Data Loading and Exploration
# ---------------------------------------------------------------------------
mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
df = pd.DataFrame(mtcars)

print("=" * 50)
print("  DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")

# ---------------------------------------------------------------------------
# 2. Correlation Analysis
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
corr_matrix = df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="BuPu",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
)
plt.title("Correlation Heatmap -- mtcars Dataset", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()

# Interpretation:
# The heatmap reveals strong multicollinearity.
# - cyl and disp: r ~ 0.90  (both measure engine size)
# - disp and wt:  r ~ 0.89  (heavier vehicles have larger engines)
# - mpg vs wt:    r ~ -0.87 (heavier cars consume more fuel)
# - mpg vs disp:  r ~ -0.85
# - mpg vs cyl:   r ~ -0.85
# This justifies the use of Ridge and Lasso regularization.

# ---------------------------------------------------------------------------
# 3. Data Preprocessing
# ---------------------------------------------------------------------------
features = df.columns[1:]  # All columns except 'mpg'
target = df.columns[0]     # 'mpg'

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print("Feature scaling applied (mean=0, std=1 on training set).")

# ---------------------------------------------------------------------------
# 4. Model Building
# ---------------------------------------------------------------------------

# 4.1 OLS Regression
X_train_ols = sm.add_constant(X_train_scaled)
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols).fit()
y_pred_ols = ols_model.predict(X_test_ols)

print("\n" + "=" * 50)
print("  OLS MODEL SUMMARY")
print("=" * 50)
print(ols_model.summary())

# 4.2 Ridge Regression (L2)
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_cv.predict(X_test_scaled)

print(f"\nOptimal Alpha (Ridge): {ridge_cv.alpha_:.4f}")

# 4.3 Lasso Regression (L1)
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_cv.predict(X_test_scaled)

print(f"Optimal Alpha (Lasso): {lasso_cv.alpha_:.4f}")

# ---------------------------------------------------------------------------
# 5. Model Comparison
# ---------------------------------------------------------------------------
results = pd.DataFrame(
    {
        "Model": ["OLS", "Ridge", "Lasso"],
        "Test R2": [
            r2_score(y_test, y_pred_ols),
            r2_score(y_test, y_pred_ridge),
            r2_score(y_test, y_pred_lasso),
        ],
        "Test MSE": [
            mean_squared_error(y_test, y_pred_ols),
            mean_squared_error(y_test, y_pred_ridge),
            mean_squared_error(y_test, y_pred_lasso),
        ],
    }
)

results["Test R2"] = results["Test R2"].round(4)
results["Test MSE"] = results["Test MSE"].round(4)

print("\n" + "=" * 50)
print("  MODEL COMPARISON -- TEST SET")
print("=" * 50)
print(results.to_string(index=False))
print("=" * 50)

# Interpretation:
# Ridge achieves the best R2 (0.8181) and lowest MSE (7.27).
# Lasso improves upon OLS while using only 3 features.
# OLS is weakest due to multicollinearity.

# ---------------------------------------------------------------------------
# 6. Lasso Feature Selection
# ---------------------------------------------------------------------------
lasso_coefs = pd.Series(lasso_cv.coef_, index=features)

num_eliminated = np.sum(lasso_coefs == 0)
print(f"\nFeatures eliminated by Lasso: {num_eliminated} out of {len(features)}")
print(f"\nLasso Coefficients:\n{lasso_coefs}")

# Feature importance bar plot
sorted_coefs = lasso_coefs.sort_values(key=abs, ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4a90d9" if c != 0 else "#cccccc" for c in sorted_coefs.values]
ax.barh(sorted_coefs.index, sorted_coefs.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Coefficient Value (Standardized)", fontsize=12)
ax.set_title("Lasso Feature Importance", fontsize=14, fontweight="bold", pad=12)
ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# Interpretation:
# Lasso eliminated 7 of 10 features. Retained predictors:
# - wt  (weight): strongest negative impact on mpg
# - cyl (cylinders): more cylinders -> lower mpg
# - hp  (horsepower): higher hp -> lower mpg
# Redundant features (disp, carb) correlated with retained ones were dropped.

# ---------------------------------------------------------------------------
# 7. Coefficient Comparison
# ---------------------------------------------------------------------------
comparison = pd.DataFrame(
    {
        "OLS": ols_model.params[1:],  # Exclude intercept
        "Ridge": ridge_cv.coef_,
        "Lasso": lasso_cv.coef_,
    },
    index=features,
)

print("\n" + "=" * 55)
print("  COEFFICIENT COMPARISON (Standardized Features)")
print("=" * 55)
print(comparison.round(4).to_string())
print("=" * 55)

# Grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(features))
width = 0.25

ax.bar(x - width, comparison["OLS"], width, label="OLS", color="#e74c3c", alpha=0.85)
ax.bar(x, comparison["Ridge"], width, label="Ridge", color="#3498db", alpha=0.85)
ax.bar(x + width, comparison["Lasso"], width, label="Lasso", color="#2ecc71", alpha=0.85)

ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Coefficient Value", fontsize=12)
ax.set_title("Coefficient Comparison: OLS vs Ridge vs Lasso", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")
ax.legend(frameon=True, fontsize=11)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# Interpretation:
# OLS coefficients are large and unstable (e.g., wt ~ -4.65, disp ~ +2.16)
#   due to multicollinearity.
# Ridge shrinks all coefficients toward zero (wt: -4.65 -> ~-0.97)
#   while retaining every feature.
# Lasso performs shrinkage + selection: only wt, cyl, hp survive.
#   All other coefficients are exactly zero.

# ---------------------------------------------------------------------------
# 8. Visual Performance Comparison
# ---------------------------------------------------------------------------
r2_scores = [
    r2_score(y_test, y_pred_ols),
    r2_score(y_test, y_pred_ridge),
    r2_score(y_test, y_pred_lasso),
]
mse_scores = [
    mean_squared_error(y_test, y_pred_ols),
    mean_squared_error(y_test, y_pred_ridge),
    mean_squared_error(y_test, y_pred_lasso),
]
model_names = ["OLS", "Ridge", "Lasso"]
colors_palette = ["#e74c3c", "#3498db", "#2ecc71"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R2 comparison
bars1 = axes[0].bar(model_names, r2_scores, color=colors_palette, edgecolor="white", linewidth=1.5)
axes[0].set_ylabel("Test R-squared", fontsize=12)
axes[0].set_title("Model Comparison -- Test R2", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, 1)
for bar, score in zip(bars1, r2_scores):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.02,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# MSE comparison
bars2 = axes[1].bar(model_names, mse_scores, color=colors_palette, edgecolor="white", linewidth=1.5)
axes[1].set_ylabel("Test MSE", fontsize=12)
axes[1].set_title("Model Comparison -- Test MSE", fontsize=13, fontweight="bold")
for bar, score in zip(bars2, mse_scores):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.15,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Regularized Regression -- Performance Summary", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Lasso feature importance (horizontal bar)
plt.figure(figsize=(8, 5))
plt.barh(features, lasso_cv.coef_, color="#4a90d9", edgecolor="white", linewidth=0.5)
plt.xlabel("Coefficient Value", fontsize=12)
plt.title("Lasso Feature Importance", fontsize=14, fontweight="bold", pad=12)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")