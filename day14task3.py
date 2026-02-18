import numpy as np
import pandas as pd

np.random.seed(0)

# Non-linear relationship: y = x^2 + noise
X = np.linspace(-5, 5, 50).reshape(-1, 1)
y = X[:, 0]**2 + np.random.normal(0, 3, 50)

df = pd.DataFrame({"X": X.flatten(), "y": y})
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_linear = lin_reg.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2
)

poly_reg = LinearRegression()
poly_reg.fit(X_train_p, y_train)

y_pred_poly = poly_reg.predict(X_test_p)
r2_poly = r2_score(y_test, y_pred_poly)
print("R² with Linear Features:     ", round(r2_linear, 3))
print("R² with Polynomial Features:", round(r2_poly, 3))
