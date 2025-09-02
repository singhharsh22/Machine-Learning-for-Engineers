import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('pendulum_data.csv')

# Ensure your dataset has the expected columns: 'theta', 'theta_dot', 'theta_double_dot'
print(data.head())

# Data Preprocessing
# Separate features and target variable
X = data[['theta', 'theta_dot']]
y = data['theta_double_dot']

# Create additional features to expand the hypothesis space
X['sin_theta'] = np.sin(X['theta'])
X['theta_dot_squared'] = X['theta_dot'] ** 2

# Rename features with standard characters
X = X.rename(columns={
    'theta': 'theta',
    'theta_dot': 'theta_dot',
    'theta_double_dot': 'theta_double_dot',
    'sin_theta': 'sin_theta',
    'theta_dot_squared': 'theta_dot_squared',
})

# Combine the features and target variable for correlation matrix calculation
data_expanded = pd.concat([X, y.rename('theta_double_dot')], axis=1)

# Exploratory Data Analysis (EDA)
correlation_matrix = data_expanded.corr()
print(correlation_matrix)

# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Set x-axis labels to be horizontal
plt.xticks(rotation=0)

plt.title('Correlation Matrix')
plt.tight_layout(pad=1.0)
plt.show()

# Scatter plot of the hypothesis space
sns.pairplot(data_expanded, diag_kind='kde')
plt.suptitle('Scatter Plot of Hypothesis Space')
plt.tight_layout(pad=3.0)
plt.show()

# Model Selection
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Ridge regression with cross-validation to find the optimal alpha
alphas = np.arange(0, 2, 0.001).tolist()
best_alpha = None
best_score = float('inf')
alpha_mse_values = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mean_score = -scores.mean()
    alpha_mse_values.append(mean_score)
    
    if mean_score < best_score:
        best_score = mean_score
        best_alpha = alpha

print(f'Best alpha: {best_alpha}')

# Plot MSE vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, alpha_mse_values, label='MSE vs Alpha')
plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Optimal Alpha: {best_alpha:.3f}')
plt.axhline(y=best_score, color='g', linestyle='--', label=f'Min MSE: {best_score:.7f}')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs Alpha for Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()

# Train the model with the best alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)

# Model Evaluation
y_pred = ridge_model.predict(X_test)

# Plot predictions against actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values (theta_double_dot)')
plt.ylabel('Predicted Values (theta_double_dot)')
plt.title('Predicted vs Actual Angular Acceleration')
plt.grid()
plt.show()

# Display coefficients and MSE
print(f'Coefficients: {ridge_model.coef_}')
mse = mean_squared_error(y_test, y_pred)
print(f'Cross-validated MSE: {mse}')

# Include the model representation using ASCII characters
print(f'Model: theta_double_dot = {ridge_model.coef_[0]:.4f} * theta + {ridge_model.coef_[1]:.4f} * theta_dot + {ridge_model.coef_[2]:.4f} * sin(theta) + {ridge_model.coef_[3]:.4f} * theta_dot_squared')
