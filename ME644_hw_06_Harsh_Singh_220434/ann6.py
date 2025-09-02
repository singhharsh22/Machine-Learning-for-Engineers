import numpy as np
import matplotlib.pyplot as plt

# Define the function to approximate
def f(x1, x2):
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

# Generate training data
def generate_data(num_samples):
    x1 = np.random.uniform(-1, 1, num_samples)
    x2 = np.random.uniform(-1, 1, num_samples)
    y = f(x1, x2)
    return np.column_stack((x1, x2)), y

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the ANN class
class ANN:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, learning_rate=0.001):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.1
        self.b1 = np.random.rand(1, hidden_size1) * 0.1
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        self.b2 = np.random.rand(1, hidden_size2) * 0.1
        self.W3 = np.random.randn(hidden_size2, hidden_size3) * 0.1
        self.b3 = np.random.rand(1, hidden_size3) * 0.1
        self.W4 = np.random.randn(hidden_size3, output_size) * 0.1
        self.b4 = np.random.rand(1, output_size) * 0.1
        self.learning_rate = learning_rate

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        return self.z4  # No activation on output layer for regression

    def backpropagation(self, X, y):
        m = X.shape[0]

        # Forward pass
        output = self.feedforward(X)

        # Compute cost
        cost = np.mean((output - y.reshape(-1, 1)) ** 2)

        # Backward pass
        d_output = 2 * (output - y.reshape(-1, 1)) / m
        dW4 = np.dot(self.a3.T, d_output)
        db4 = np.sum(d_output, axis=0, keepdims=True)

        d_a3 = np.dot(d_output, self.W4.T) * sigmoid_derivative(self.z3)
        dW3 = np.dot(self.a2.T, d_a3)
        db3 = np.sum(d_a3, axis=0, keepdims=True)

        d_a2 = np.dot(d_a3, self.W3.T) * sigmoid_derivative(self.z2)
        dW2 = np.dot(self.a1.T, d_a2)
        db2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, d_a1)
        db1 = np.sum(d_a1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4

        return cost

    def predict(self, X):
        return self.feedforward(X)

    def compute_test_error(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.mean((predictions - y_test.reshape(-1, 1)) ** 2)

    def train(self, X, y, X_test, y_test, epochs=10000):
        costs = []
        train_relative_errors = []
        test_relative_errors = []

        for epoch in range(epochs):
            cost = self.backpropagation(X, y)
            costs.append(cost)

            # Print cost every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost:.4f}")

            # Calculate training and test predictions
            train_predictions = self.predict(X)
            test_predictions = self.predict(X_test)

            # Calculate absolute errors
            train_absolute_error = np.abs(train_predictions - y.reshape(-1, 1))
            test_absolute_error = np.abs(test_predictions - y_test.reshape(-1, 1))

            # Calculate mean absolute errors
            mean_train_error = np.mean(train_absolute_error)
            mean_test_error = np.mean(test_absolute_error)

            # Calculate relative error percentages
            train_relative_error_percentage = (mean_train_error / np.mean(y)) * 100
            test_relative_error_percentage = (mean_test_error / np.mean(y_test)) * 100

            # Append relative errors to their respective lists
            train_relative_errors.append(train_relative_error_percentage)
            test_relative_errors.append(test_relative_error_percentage)

        return costs, train_relative_errors, test_relative_errors  # Return both errors

# Generate training and testing data
X_train, y_train = generate_data(200)
X_test, y_test = generate_data(100)

# Initialize and train the ANN
ann = ANN(input_size=2, hidden_size1=25, hidden_size2=25, hidden_size3=25, output_size=1, learning_rate=0.001)
costs, train_relative_errors, test_relative_errors = ann.train(X_train, y_train, X_test, y_test, epochs=10000)

# Print final Mean Relative Error Percentages
print(f"\nFinal Mean Relative Error Percentage for Training: {train_relative_errors[-1]:.4f}%")
print(f"Final Mean Relative Error Percentage for Testing: {test_relative_errors[-1]:.4f}%")

# Print weights and biases
print("\nWeights and Biases:")
print(f"W1:\n{ann.W1}\nb1:\n{ann.b1}\n")
print(f"W2:\n{ann.W2}\nb2:\n{ann.b2}\n")
print(f"W3:\n{ann.W3}\nb3:\n{ann.b3}\n")
print(f"W4:\n{ann.W4}\nb4:\n{ann.b4}\n")

# Plot the cost function
plt.figure(figsize=(10, 5))
plt.plot(costs, color='blue')
plt.title('Cost Function Over Epochs')
plt.xlabel('Epochs (per 1)')
plt.ylabel('Cost')
plt.grid()
plt.show()

# Plot Mean Relative Error Percentage
plt.figure(figsize=(10, 5))
plt.plot(train_relative_errors, color='red', label='Training Mean Relative Error %')
plt.plot(test_relative_errors, color='green', label='Testing Mean Relative Error %')
plt.title('Mean Relative Error Percentage Over Epochs')
plt.xlabel('Epochs (per 1)')
plt.ylabel('Mean Relative Error Percentage (%)')
plt.legend()
plt.grid()
plt.show()
