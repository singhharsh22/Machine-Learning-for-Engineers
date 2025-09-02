import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split  # Include train_test_split here
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('pendulum_data.csv')
theta = data['theta'].values
theta_dot = data['theta_dot'].values
theta_double_dot = data['theta_double_dot'].values

# Exploratory Data Analysis
# Calculate correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)

# Scatter plots
plt.figure(figsize=(12, 8))
sns.pairplot(data)
plt.show()

# Create features for hypothesis space
X = np.column_stack((theta, np.sin(theta), theta_dot, theta_dot**2))
y = theta_double_dot

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regression with cross-validation
ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_results=True)
ridge_model.fit(X_train, y_train)

# Final model parameters
print("Best alpha:", ridge_model.alpha_)
print("Coefficients:", ridge_model.coef_)

# Cross-validation
kf = KFold(n_splits=5)
mse_list = []
for train_index, val_index in kf.split(X):
    X_train_kf, X_val_kf = X[train_index], X[val_index]
    y_train_kf, y_val_kf = y[train_index], y[val_index]
    ridge_model.fit(X_train_kf, y_train_kf)
    mse = mean_squared_error(y_val_kf, ridge_model.predict(X_val_kf))
    mse_list.append(mse)

print("Cross-validated MSE:", np.mean(mse_list))

# Plotting predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_model.predict(X_test))
plt.plot([-10, 10], [-10, 10], 'r--')  # Line of equality
plt.xlabel('Actual $\\ddot{\\theta}$')
plt.ylabel('Predicted $\\ddot{\\theta}$')
plt.title('Actual vs Predicted Angular Acceleration')
plt.show()

import matplotlib.pyplot as plt

# Assuming you have y_test and y_pred from your predictions
y_pred = ridge_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()
