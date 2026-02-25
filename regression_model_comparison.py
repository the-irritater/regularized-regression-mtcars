import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')
warn()

mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
df = pd.DataFrame(mtcars)

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="BuPu", vmin=-1, vmax=1)
plt.title("Correlation Heatmap - mtcars")
plt.show()

"""The correlation heatmap revealed strong multicollinearity among predictors. For example:



*   Cylinders (cyl) and displacement (disp) show a very high positive correlation (~0.90).
*   mpg has strong negative correlations with:

    weight (wt) ≈ -0.87

    displacement (disp) ≈ -0.85

    cylinders (cyl) ≈ -0.85
This indicates that heavier cars with larger engines tend to have lower fuel efficiency. The presence of multicollinearity justifies the use of regularization techniques such as Ridge and Lasso regression.
"""

features = df.columns[1:]
target = df.columns[0]
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

#scaling and centering the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#OLS
X_train_ols = sm.add_constant(X_train_scaled)
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols).fit()
y_pred_ols = ols_model.predict(X_test_ols)

# Ridge
ridge_cv = RidgeCV(alphas=np.logspace(-4,4,100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_cv.predict(X_test_scaled)

# Lasso
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_cv.predict(X_test_scaled)

results = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso"],
    "Test R2": [
        r2_score(y_test, y_pred_ols),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_lasso)
    ],
    "Test MSE": [mean_squared_error(y_test, y_pred_ols),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_lasso)]})

print("\nModel Comparison:\n")
print(results)

"""##OLS (Ordinary Least Squares)

The OLS model achieved a Test R² of 0.7466, meaning it explains approximately 74.66% of the variance in fuel efficiency (mpg). However, the Test MSE of 10.13 indicates relatively higher prediction error compared to regularized models.

Since OLS does not handle multicollinearity, its performance is slightly weaker when predictors are highly correlated.

##Ridge Regression

The Ridge model produced the best performance among all three models, with a Test R² of 0.8181 and a Test MSE of 7.27.


*   Ridge explains 81.81% of the variance in mpg.
*   It significantly reduces prediction error compared to OLS.

Ridge regression improves model stability by shrinking coefficients without eliminating features, making it more robust in the presence of multicollinearity.

##Lasso Regression

The Lasso model achieved a Test R² of 0.7770 and a Test MSE of 8.92.

Compared to OLS:



*   Lasso improves predictive performance.

*   It reduces MSE from 10.13 to 8.92.

Compared to Ridge:



*   Lasso performs slightly worse in prediction accuracy.
*   However, it provides the additional benefit of feature selection by shrinking some coefficients to zero.
"""

lasso_coefs = pd.Series(lasso_cv.coef_, index=features)

print("\nOptimal Alpha (Lasso):", lasso_cv.alpha_)

"""The selected alpha value (0.8918) confirms that regularization plays an important role in improving model generalization for this dataset. By applying this penalty strength, Lasso balances bias and variance while simplifying the model."""

print("\nLasso Coefficients:\n")
print(lasso_coefs)

# Eliminated features
num_zero = np.sum(lasso_coefs == 0)
print("\nNumber of features eliminated by Lasso:", num_zero)

"""After fitting the Lasso model, I observed that 7 out of 10 features were shrunk to zero, meaning the model automatically eliminated less important variables. The remaining influential predictors were weight (wt), horsepower (hp), and cylinders (cyl). This indicates that these three features are the strongest drivers of fuel efficiency when multicollinearity is taken into account."""

# Plot coefficients
lasso_coefs.sort_values(key=abs, ascending=False).plot(kind='bar', figsize=(8, 5))
plt.title("Lasso Feature Importance")
plt.ylabel("Coefficient Value")
plt.show()

"""The Lasso feature importance plot, I can clearly see that weight (wt) has the largest negative coefficient, which means it has the strongest impact on fuel efficiency. As the weight of the vehicle increases, the mpg decreases significantly. I also observe that cylinders (cyl) and horsepower (hp) have negative coefficients, indicating that vehicles with more cylinders and higher horsepower tend to consume more fuel. The remaining features were shrunk to zero, which shows that Lasso eliminated less important variables and retained only the most significant predictors affecting mpg."""

comparison = pd.DataFrame({
    "OLS": ols_model.params[1:],   # remove intercept
    "Ridge": ridge_cv.coef_,
    "Lasso": lasso_cv.coef_
}, index=features)

print("\nCoefficient Comparison:\n")
print(comparison)

"""The coefficient comparison table, I observe clear differences in how OLS, Ridge, and Lasso handle the predictors. In the OLS model, several coefficients such as wt (-4.65) and disp (2.16) have large magnitudes, which may be influenced by multicollinearity. Ridge regression shrinks all coefficients toward zero, reducing their magnitude while retaining every feature. For example, the weight coefficient is reduced from -4.65 (OLS) to -0.97 (Ridge), showing how Ridge stabilizes the model.

Lasso performs both shrinkage and feature selection. It completely eliminates several variables such as disp, drat, qsec, vs, am, gear, and carb, setting their coefficients to zero. Only wt (-2.49), cyl (-1.57), and hp (-0.64) remain significant in the Lasso model. This confirms that these three variables are the most important predictors of fuel efficiency when multicollinearity is considered.
"""