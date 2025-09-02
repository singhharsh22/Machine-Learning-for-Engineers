# Importing necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.linear_model import LinearRegression  # For creating and using linear regression models
from sklearn.metrics import r2_score  # For calculating the coefficient of determination (R²)

# Data: Temperature and Hardness values
temp = np.array([30, 30, 30, 30, 40, 40, 40, 50, 50, 50, 60, 60, 60, 60])  # Temperature data in °C
hardness = np.array([55.8, 59.1, 54.8, 54.6, 43.1, 42.2, 45.2, 31.6, 30.9, 30.8, 17.5, 20.5, 17.2, 16.9])  # Hardness data

# Reshaping the temperature array to be a 2D array with a single feature
temperature = temp.reshape(-1, 1)  # Reshape to (14, 1) for the regression model

# After reshaping, temperature array will look like this:
# array([[30],
#        [30],
#        [30],
#        [30],
#        [40],
#        [40],
#        [40],
#        [50],
#        [50],
#        [50],
#        [60],
#        [60],
#        [60],
#        [60]])


# Create and fit the linear regression model
model = LinearRegression()  # Instantiate the LinearRegression model
model.fit(temperature, hardness)  # Fit the model using the temperature and hardness data

# Make predictions using the fitted model
predicted_hardness = model.predict(temperature)  # Predict hardness values based on the model

# Retrieve the model's coefficients and intercept
print(f"Coefficient (Bias): {model.coef_[0]}")  # Print the model's coefficient (slope)
print(f"Intercept (Weight): {model.intercept_}")  # Print the model's intercept (y-intercept)

# Calculate the coefficient of determination (R²) to evaluate the model's performance
r_squared = r2_score(hardness, predicted_hardness)  # Compute R² score

# Format each predicted hardness value to 4 decimal places for clearer output
formatted_predictions = ", ".join([f"{pred:.4f}" for pred in predicted_hardness])  # Format predictions

# Print the predicted hardness values and R² score
#print(f"Predicted Hardness: {predicted_hardness}")  # Original print statement
print(f"Predicted Hardness: {formatted_predictions}")  # Print formatted predictions
print(f"R² Score: {r_squared:.4f}")  # Print R² score to 4 decimal places

# Plot actual vs. predicted hardness
plt.figure(figsize=(10, 6))  # Create a new figure with specified size
plt.scatter(temperature, hardness, color='blue', label='Actual Hardness')  # Plot actual data points
plt.plot(temperature, predicted_hardness, color='red', linewidth=2, label='Fitted Line')  # Plot the fitted regression line
plt.xlabel('Temperature')  # X-axis label
plt.ylabel('Hardness')  # Y-axis label
plt.title(f'Predicted Hardness vs Actual Hardness')  # Title of the plot
plt.legend()  # Show legend for the plot
plt.grid(True)  # Add grid lines for better readability
plt.show()  # Display the plot

# # Plotting the data and the linear hypothesis
# plt.figure(figsize=(10, 6))  # Create another figure with specified size
# plt.scatter(temperature, hardness, marker='x', color='blue', label='Observed Data')  # Plot actual data with 'x' markers
# plt.plot(temperature, predicted_hardness, color='red', label='Linear Hypothesis')  # Plot the linear hypothesis line
# plt.xlabel('Temperature (°C)')  # X-axis label with units
# plt.ylabel('Hardness')  # Y-axis label
# plt.title('Linear Hypothesis of Hardness vs Temperature')  # Title of the plot
# plt.legend()  # Show legend for the plot
# plt.grid(True)  # Add grid lines for better readability
# plt.show()  # Display the plot

# Display the R² value again for emphasis
print(f'Coefficient of Determination (R²): {r_squared:.4f}')  # Print R² score to 4 decimal places