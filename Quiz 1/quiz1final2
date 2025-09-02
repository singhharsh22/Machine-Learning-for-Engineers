import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

# Data
year = np.array([1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008])
time = np.array([10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85, 9.69])

# Normalize the data between 0 and 1
year_norm = (year - year.min()) / (year.max() - year.min())
time_norm = (time - time.min()) / (time.max() - time.min())

# Reshape year to a 2D array with 1 column
year_norm = year_norm.reshape(-1, 1)

# Linear Regression
model_linear = LinearRegression()
model_linear.fit(year_norm, time_norm)
predicted_linear_norm = model_linear.predict(year_norm)

# Recover original scale for linear regression
predicted_linear = predicted_linear_norm * (time.max() - time.min()) + time.min()

# Polynomial Regression (Degree 2)
year_squared_norm = year_norm ** 2
X2_norm = np.hstack((year_norm, year_squared_norm))

model_poly2 = LinearRegression()
model_poly2.fit(X2_norm, time_norm)
predicted_poly2_norm = model_poly2.predict(X2_norm)

# Recover original scale for polynomial regression (degree 2)
predicted_poly2 = predicted_poly2_norm * (time.max() - time.min()) + time.min()

# Polynomial Regression (Degree 4)
year_cubed_norm = year_norm ** 3
year_quadrupled_norm = year_norm ** 4
X4_norm = np.hstack((X2_norm, year_cubed_norm, year_quadrupled_norm))

model_poly4 = LinearRegression()
model_poly4.fit(X4_norm, time_norm)
predicted_poly4_norm = model_poly4.predict(X4_norm)

# Recover original scale for polynomial regression (degree 4)
predicted_poly4 = predicted_poly4_norm * (time.max() - time.min()) + time.min()

# Plotting
plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(year, time, color='blue', label='Actual Time')

# Plot linear regression line
plt.plot(year, predicted_linear, color='green', linewidth=2, label=f'Linear Regression (R² = {r2_score(time, predicted_linear):.2f})')

# Plot degree 2 polynomial regression line
plt.plot(year, predicted_poly2, color='orange', linewidth=2, label=f'Polynomial Regression (Degree 2, R² = {r2_score(time, predicted_poly2):.2f})')

# Plot degree 4 polynomial regression line
plt.plot(year, predicted_poly4, color='red', linewidth=2, label=f'Polynomial Regression (Degree 4, R² = {r2_score(time, predicted_poly4):.2f})')

# X-axis label
plt.xlabel('Year')

# Y-axis label
plt.ylabel('Best 100m Time (seconds)')

# Title of the plot
plt.title('Original Data with Linear and Polynomial Regression (Degree 2 & Degree 4)')

# Show legend for the plot
plt.legend()

# Add grid lines for better readability
plt.grid(True)

# Display the plot
plt.show()

# Define LOOCV function
def loocv(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test[0])
    return r2_score(y_true, y_pred)

# Perform LOOCV
r2_linear_loocv = loocv(model_linear, year_norm, time_norm)
r2_poly2_loocv = loocv(model_poly2, X2_norm, time_norm)
r2_poly4_loocv = loocv(model_poly4, X4_norm, time_norm)

# Print LOOCV results
print(f'LOOCV R² for Linear Regression: {r2_linear_loocv:.2f}')
print(f'LOOCV R² for Polynomial Regression (Degree 2): {r2_poly2_loocv:.2f}')
print(f'LOOCV R² for Polynomial Regression (Degree 4): {r2_poly4_loocv:.2f}')
