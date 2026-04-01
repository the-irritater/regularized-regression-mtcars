# Regularized Regression Analysis on the mtcars Dataset

A comparative study of OLS, Ridge, and Lasso regression models for predicting fuel efficiency (mpg) using the mtcars dataset, with emphasis on handling multicollinearity through regularization techniques.

---

| **Field**        | **Details**                                |
|------------------|-------------------------------------------|
| **Author**       | Sanman Kadam                              |
| **Email**        | sanman.kadam@statistics.mu.ac.in          |
| **Date**         | April 2026                                |
| **Dataset**      | mtcars (R datasets, 32 observations)      |

---

## Problem Statement

In automotive engineering and environmental policy, understanding the factors that influence fuel efficiency is critical for designing vehicles that minimize fuel consumption and reduce emissions. The **mtcars** dataset captures 10 mechanical and design attributes for 32 automobiles, many of which are highly correlated with one another (multicollinearity). When standard regression techniques such as Ordinary Least Squares (OLS) are applied to such data, the resulting coefficient estimates become **unstable and unreliable**, leading to poor predictive performance and misleading interpretations of feature importance.

The central question is: **How can we build a regression model that accurately predicts fuel efficiency (mpg) while handling multicollinearity among the predictor variables?**

---

## Objectives

1. Build and evaluate an **OLS regression model** as a baseline for predicting miles per gallon (mpg) from 10 vehicle characteristics.
2. Apply **Ridge Regression (L2 regularization)** to stabilize coefficient estimates and improve prediction accuracy in the presence of multicollinearity.
3. Apply **Lasso Regression (L1 regularization)** to perform automatic feature selection and identify the most influential predictors of fuel efficiency.
4. Compare all three models quantitatively using **R-squared** and **Mean Squared Error (MSE)** on a held-out test set.
5. Analyze and interpret the **coefficient behavior** across OLS, Ridge, and Lasso to demonstrate how regularization addresses multicollinearity.
6. Identify the **key vehicle attributes** that most strongly drive fuel efficiency.

---

## Project Overview

This project investigates how **regularization techniques** address multicollinearity in regression models. Using the mtcars dataset, we compare:

- **Ordinary Least Squares (OLS)** -- baseline model with no regularization
- **Ridge Regression (L2)** -- shrinks coefficients to stabilize estimates
- **Lasso Regression (L1)** -- performs automatic feature selection via sparsity

---

## Model Performance (Test Set)

| Model  | Test R-squared | Test MSE | Features Used |
|--------|----------------|----------|---------------|
| OLS    | 0.7466         | 10.13    | 10            |
| Ridge  | 0.8181         | 7.27     | 10            |
| Lasso  | 0.7770         | 8.91     | 3             |

**Ridge Regression** achieved the best predictive performance. **Lasso Regression** provided the most interpretable model by retaining only 3 of the 10 features.

---

## Key Findings

- **Optimal Alpha (Lasso):** 0.8918
- **Lasso eliminated 7 out of 10 features**, retaining only:
  - Weight (`wt`) -- strongest negative impact on fuel efficiency
  - Horsepower (`hp`) -- higher power reduces mpg
  - Cylinders (`cyl`) -- more cylinders lower mpg
- **Ridge improved R-squared by approximately 10 percentage points** over OLS by stabilizing coefficient estimates
- **Multicollinearity** between `cyl`, `disp`, and `wt` (r > 0.85) confirmed the need for regularization

---

## Repository Structure

```
.
├── Regularized_Regression_Analysis.ipynb   # Jupyter notebook with full analysis and interpretations
├── regression_model_comparison.py          # Standalone Python script
├── requirements.txt                        # Python dependencies
├── Images/                                 # Plots and figures
└── README.md                               # This file
```

---

## Skills Demonstrated

- Data Exploration and Preprocessing
- Feature Scaling (StandardScaler)
- Regularization Techniques (Ridge L2, Lasso L1)
- Hyperparameter Tuning via Cross-Validation
- Model Evaluation (R-squared, MSE)
- Automated Feature Selection
- Statistical Interpretation

---

## Technologies Used

| Library       | Purpose                          |
|---------------|----------------------------------|
| NumPy         | Numerical computing              |
| Pandas        | Data manipulation                |
| Scikit-learn  | Ridge, Lasso, model evaluation   |
| Statsmodels   | OLS regression, dataset loading  |
| Matplotlib    | Visualization                    |
| Seaborn       | Statistical plots and styling    |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Python script
python regression_model_comparison.py

# Or open the Jupyter notebook
jupyter notebook Regularized_Regression_Analysis.ipynb
```

---

## License

MIT
