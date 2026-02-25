# Regularized Regression Analysis on Auto MPG Dataset

## Project Overview
This project compares OLS, Ridge, and Lasso regression models to predict fuel efficiency (mpg) using the mtcars dataset.

The objective is to analyze how regularization improves model performance and handles multicollinearity.

---

## Models Implemented
- Ordinary Least Squares (OLS)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

---

## Model Performance (Test Set)

| Model  | Test R² | Test MSE |
|--------|---------|----------|
| OLS    | 0.7466  | 10.13    |
| Ridge  | 0.8181  | 7.27     |
| Lasso  | 0.7770  | 8.91     |

Ridge Regression achieved the best performance.

---

## Key Findings

- Optimal Alpha (Lasso): 0.8918
- Lasso shrunk 7 out of 10 features to zero
- Most influential predictors:
  - Weight (wt)
  - Horsepower (hp)
  - Cylinders (cyl)

Weight has the strongest negative impact on fuel efficiency.

---

## Skills Demonstrated

- Data Preprocessing
- Feature Scaling
- Regularization Techniques
- Cross Validation
- Model Evaluation (R², MSE)
- Feature Selection

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
python regularized_regression_mtcars.py
