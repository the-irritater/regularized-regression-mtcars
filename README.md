# Regularized Regression Analysis on mtcars Dataset

Comparative evaluation of Ordinary Least Squares (OLS), Ridge (L2 penalty), Lasso (L1 penalty), and ElasticNet regression models for predicting fuel efficiency (MPG).

## Problem Statement

Multicollinearity among vehicle engine specifications (displacement, horsepower, cylinder count, weight) destabilizes OLS parameter estimates, leading to inflated variance.

This project applies regularized regression techniques to stabilize coefficient estimation and perform automated feature selection.

## Model Evaluation & Coefficient Shrinkage

### Regression Model Comparison

| Regression Model | Penalty Constraint | Test RMSE | Adjusted R-Squared | Zeroed Coefficients | Key Selected Features |
|---|---|---|---|---|---|
| OLS Linear Regression | None | 3.24 | 0.812 | 0 | All features retained |
| Ridge Regression | L2 ($\alpha = 1.5$) | 2.85 | 0.845 | 0 | Shrinks all weights smoothly |
| **Lasso Regression** | **L1 ($\alpha = 0.4$)** | **2.68** | **0.861** | **3** | **wt (Weight), hp (Horsepower), qsec** |
| ElasticNet | L1 + L2 ($\alpha = 0.5, l1\_ratio = 0.5$) | 2.74 | 0.854 | 2 | Hybrid parameter selection |

## Project Structure

```
regularized-regression-mtcars/
├── Images/
│   └── coefficient_shrinkage_paths.png
├── Regularized_Regression_Analysis.ipynb
├── regression_model_comparison.py
├── trace_plot.py
├── requirements.txt
└── README.md
```

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Shrinkage Path Plots
```bash
python trace_plot.py
```

### Run Full Regression Benchmark
```bash
python regression_model_comparison.py
```

## Author

Sanman Kadam  
MSc Statistics | Data Analyst
