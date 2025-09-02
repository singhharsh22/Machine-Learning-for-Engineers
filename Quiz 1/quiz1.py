import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data
year = np.array([1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008])
time = np.array([10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85, 9.69])

# Reshape year to a 2D array with 1 column
year = year.reshape(-1, 1)

# Instantiate and fit the LinearRegression model using the combined features
model_linear = LinearRegression()
model_linear.fit(year, time)

# Predict time values using the combined features
predicted_linear = model_linear.predict(year)

# Print all the model's coefficients and intercept
print(f"Intercept: {model_linear.intercept_}")
print(f"Coefficient for Year (x): {model_linear.coef_[0]}")
print(f"Coefficient for Year^2 (x^2): {model_linear.coef_[1]}")
print(f"Coefficient for Year^3 (x^2): {model_linear.coef_[2]}")
print(f"Coefficient for Year^4 (x^2): {model_linear.coef_[3]}")

# Compute R² score
r_squared_linear = r2_score(time, predicted_linear)
print(f"R² score: {r_squared_linear}")


# Manually create the x^2 feature
year_squared = year ** 2
# Combine year and year_squared to form the feature matrix
X2 = np.hstack((year, year_squared))

# Instantiate and fit the LinearRegression model using the combined features
model_poly2 = LinearRegression()
model_poly2.fit(X2, time)

# Predict time values using the combined features
predicted_poly2 = model_poly2.predict(X2)

# Print all the model's coefficients and intercept
print(f"Intercept: {model_poly2.intercept_}")
print(f"Coefficient for Year (x): {model_poly2.coef_[0]}")
print(f"Coefficient for Year^2 (x^2): {model_poly2.coef_[1]}")

# Compute R² score
r_squared_poly2 = r2_score(time, predicted_poly2)
print(f"R² score: {r_squared_poly2}")

# Manually create the x^3 feature
year_cubed = year ** 3
# Combine X2 and year_cubed to form the feature matrix
X3 = np.hstack((X2, year_cubed))

# Manually create the year^4 feature
year_quadrupled = year ** 4
# Combine X3 and year_quadrupled to form the feature matrix
X4 = np.hstack((X3, year_quadrupled))


# Instantiate and fit the LinearRegression model using the combined features
model = LinearRegression()
model.fit(X4, time)

# Predict time values using the combined features
predicted_poly4 = model.predict(X4)

# Print all the model's coefficients and intercept
print(f"Intercept: {model.intercept_}")
print(f"Coefficient for Year (x): {model.coef_[0]}")
print(f"Coefficient for Year^2 (x^2): {model.coef_[1]}")
print(f"Coefficient for Year^3 (x^2): {model.coef_[2]}")
print(f"Coefficient for Year^4 (x^2): {model.coef_[3]}")

# Compute R² score
r_squared_poly4 = r2_score(time, predicted_poly4)
print(f"R² score (degree 4): {r_squared_poly4}")

# Create a new figure with specified size
plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(year, time, color='blue', label='Actual Time')

#Plot linear regression line
plt.plot(year, predicted_linear, color='green', linewidth=2, label=f'Linear Regression (R² = {r2_linear:.2f})')

# Plot degree 2 polynomial regression line
plt.plot(year, predicted_poly2, color='orange', linewidth=2, label=f'Polynomial Regression (Degree 2, R² = {r2_poly2:.2f})')

# Plot degree 4 polynomial regression line
plt.plot(year, predicted_poly4, color='red', linewidth=2, label=f'Polynomial Regression (Degree 4, R² = {r2_poly4:.2f})')


# Plot the multiple linear regression line
plt.plot(year, predicted_time, color='red', linewidth=2, label='Multiple Linear Regression Line')

# X-axis label
plt.xlabel('Year')

# Y-axis label
plt.ylabel('Best 100m Time (seconds)')

# Title of the plot
plt.title(f'Predicted Time vs Actual Time (Multiple Linear Regression, R² = {r_squared:.2f})')

# Show legend for the plot
plt.legend()

# Add grid lines for better readability
plt.grid(True)

# Display the plot
plt.show()
