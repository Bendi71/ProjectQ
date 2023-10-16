import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

elteres=[]

np.random.seed(1)
x = np.random.rand(100, 1)  # Random input data
e = np.random.randn(100, 1)  # Random noise
y = x ** 2 - 1.5 * x + e  # Generating target variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)
y_pred_s = model.predict(x)

# Get the model coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0][0]

# Create polynomial features (quadratic features)
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)

# Create and fit the quadratic regression model
model = LinearRegression()
model.fit(x_poly, y)

# Make predictions
y_pred = model.predict(x_poly)

# Sort the data and predictions for a smoother plot
sort_indices = np.argsort(x[:, 0])
x_sorted = x[sort_indices]
y_pred_sorted = y_pred[sort_indices]

# Get the model coefficients
int_poly = model.intercept_[0]
coef_poly = model.coef_

print("Intercept:", intercept, int_poly)
print("Coefficients:", coefficients, coef_poly)

# Make a prediction for x = 0
x_new = np.array([[0]])  # Point for prediction
x_new_poly = poly_features.transform(x_new)  # Transform the point
y_pred = model.predict(x_new_poly)  # Make the prediction

print(y_pred)
# Plot the data and regression line
plt.scatter(x, y, label='Actual Data')
plt.plot(x, y_pred_s, color='red', linewidth=3, label='Regression Line')
plt.plot(x_sorted, y_pred_sorted, color='green', linewidth=3, label='Quadratic Regression Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


