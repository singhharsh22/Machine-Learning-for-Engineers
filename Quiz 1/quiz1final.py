import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data
year = np.array([1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008])
time = np.array([10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85, 9.69])

# Reshaping temp array to a 2D array with 1 column
year=year.reshape(-1,1)

# Instantiate the LinearRegression model
model = LinearRegression()

# Fit the model using the temperature and hardness data
model.fit(year, time)

# Predict hardness values based on the model
predicted_time = model.predict(year)

# Print the model's coefficient (slope)
print(f"Coefficient (Bias): {model.coef_[0]}")

# Print the model's intercept (y-intercept)
print(f"Intercept (Weight): {model.intercept_}")

# Compute R² score
r_squared = r2_score(time, predicted_time)

# Create a new figure with specified size
plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(year, time, color='blue', label='Actual Time')
plt.plot(year, predicted_time, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Year')
plt.ylabel('Time')
plt.title(f'Predicted Time vs Actual Time')
plt.legend()
plt.grid(True)
plt.show()