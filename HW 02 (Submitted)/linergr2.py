import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data
temp = np.array([30, 30, 30, 30, 40, 40, 40, 50, 50, 50, 60, 60, 60, 60])
hardness = np.array([55.8, 59.1, 54.8, 54.6, 43.1, 42.2, 45.2, 31.6, 30.9, 30.8, 17.5, 20.5, 17.2, 16.9])

# Reshaping temp array to a 2D array with 1 column
temperature=temp.reshape(-1,1)

"""
After reshaping, temperature will look like this:

array([[30],
       [30],
       [30],
       [30],
       [40],
       [40],
       [40],
       [50],
       [50],
       [50],
       [60],
       [60],
       [60],
       [60]])
"""

# Create and fit the model
model = LinearRegression()
model.fit(temperature, hardness)

# Make predictions
predicted_hardness = model.predict(temperature)

# Get the model's coefficients and intercept
# model.coef_ is an array
print(f"Coefficient (Bias): {model.coef_[0]}")  # [0] is used to extract the value from the array
print(f"Intercept (Weight): {model.intercept_}")

# Coefficient of determination (R²)
r_squared = r2_score(hardness, predicted_hardness)

# Format each predicted hardness value to 4 decimal places
formatted_predictions = ", ".join([f"{pred:.4f}" for pred in predicted_hardness])

# Print results
#print(f"Predicted Hardness: {predicted_hardness}")
print(f"Predicted Hardness: {formatted_predictions}")
print(f"R² Score: {r_squared:.4f}")

# Plot actual vs. predicted hardness
plt.figure(figsize=(10, 6))
plt.scatter(temperature, hardness, color='blue', label='Actual Hardness')
plt.plot(temperature, predicted_hardness, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Temperature')
plt.ylabel('Hardness')
plt.title(f'Predicted Hardness vs Actual Hardness')
plt.legend()
plt.grid(True)
plt.show()

print(f"R² Score: {r_squared:.4f}")

# Plotting the data and the linear hypothesis

# Plotting actual vs. predicted hardness
plt.figure(figsize=(10, 6))
plt.scatter(temperature, hardness, marker='x', color='blue', label='Observed Data')
plt.plot(temperature, predicted_hardness, color='red', label='Linear Hypothesis')
plt.xlabel('Temperature (°C)')
plt.ylabel('Hardness')
plt.title('Linear Hypothesis of Hardness vs Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Display the R² value
print(f'Coefficient of Determination (R²): {r_squared:.4f}')