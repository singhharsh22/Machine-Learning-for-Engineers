import numpy as np
import matplotlib.pyplot as plt

X = np.array([[15.5,40], [23.75,23.25], [8,17], [17,21], [5.5,10], 
              [19,12], [24,20], [2.5,12], [7.5,15], [11,26]])
y = np.array([5,5,2,5,2,2,5,2,2,5])

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

weights = np.random.rand(X.shape[1])
bias = np.random.rand()
learning_rate = 0.001
epochs = 1000

errors = []

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X_normalized)):
        linear_output = np.dot(X_normalized[i], weights) + bias
        
        # Binary classification threshold
        y_pred = 5 if linear_output > 3.5 else 2
        
        error = y[i] - y_pred
        if error != 0:
            weights += learning_rate * error * X_normalized[i]
            bias += learning_rate * error
            total_error += 1
    errors.append(total_error)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Total Misclassification Error = {total_error}")

x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
Z = Z.reshape(xx.shape)

xx_original = xx * (X_max[0] - X_min[0]) + X_min[0]
yy_original = yy * (X_max[1] - X_min[1]) + X_min[1]

plt.figure(figsize=(10, 6))

Z_class = np.where(Z > 3.5, 5, 2)

contour = plt.contourf(xx_original, yy_original, Z_class, levels=[1.5, 2.5, 5.5], colors=['red', 'blue'], alpha=0.8)

plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='lightcoral', label='Class 2')
plt.scatter(X[y == 5][:, 0], X[y == 5][:, 1], color='green', label='Class 5')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary on Original Data')
plt.legend()

cbar = plt.colorbar(contour)
cbar.set_ticks([2, 5])
cbar.set_label("Class Prediction")

plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(epochs), errors, color='purple')
plt.xlabel('Epoch')
plt.ylabel('Total Misclassification Error')
plt.title('Error Reduction Over Epochs')
plt.show()
