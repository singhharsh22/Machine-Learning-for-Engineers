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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3  # No activation on output layer for regression

    def backpropagation(self, X, y, dropout_rate=0.01):
        m = X.shape[0]

        # Forward pass
        output = self.feedforward(X)

        # Compute cost
        cost = np.mean((output - y.reshape(-1, 1)) ** 2)

        # Backward pass
        d_output = 2 * (output - y.reshape(-1, 1)) / m
        dW3 = np.dot(self.a2.T, d_output)
        db3 = np.sum(d_output, axis=0, keepdims=True)

        d_a2 = np.dot(d_output, self.W3.T) * sigmoid_derivative(self.z2)
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

        return cost

    def train(self, X, y, epochs=100000, dropout_rate=0.01):
        costs = []
        for epoch in range(epochs):
            cost = self.backpropagation(X, y, dropout_rate)
            costs.append(cost)
            if epoch % 10000 == 0:
                print(f'Epoch {epoch}, Cost: {cost}')
        return costs

    def predict(self, X):
        return self.feedforward(X)

    def compute_test_error(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y.reshape(-1, 1)) ** 2)

# Generate training and test data
X_train, y_train = generate_data(200)
X_test, y_test = generate_data(100)

# Initialize and train the ANN
ann = ANN(input_size=2, hidden_size1=10, hidden_size2=10, output_size=1, learning_rate=0.01)
costs = ann.train(X_train, y_train, epochs=100000)

# Calculate training and test errors
train_error = np.mean((ann.predict(X_train) - y_train.reshape(-1, 1)) ** 2)
test_error = ann.compute_test_error(X_test, y_test)

print(f'Training Error: {train_error}')
print(f'Test Error: {test_error}')

# Plot the cost function
plt.plot(costs)
plt.title('Cost Function Over Epochs')
plt.xlabel('Epochs (per 10000)')
plt.ylabel('Cost')
plt.show()
