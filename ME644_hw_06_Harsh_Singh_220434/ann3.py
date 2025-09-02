import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Forward propagation with batch normalization
def forward_propagation(X):
    global weights_input_to_hidden, biases_hidden, weights_hidden_to_output, biases_output

    # Input to hidden layer
    u = np.dot(X, weights_input_to_hidden) + biases_hidden
    u_norm = (u - np.mean(u, axis=0)) / (np.std(u, axis=0) + 1e-8)  # Batch normalization
    z = sigmoid(u_norm)

    # Hidden to output layer
    v = np.dot(z, weights_hidden_to_output) + biases_output
    yhat = sigmoid(v)
    
    return z, yhat

# Backpropagation
def backpropagation(X, Y, z, yhat):
    e = yhat - Y
    dw_output = np.dot(z.T, e)
    db_output = np.sum(e, axis=0, keepdims=True)

    dz_hidden = np.dot(e, weights_hidden_to_output.T) * sigmoid_derivative(z)
    dw_hidden = np.dot(X.T, dz_hidden)
    db_hidden = np.sum(dz_hidden, axis=0, keepdims=True)

    return dw_hidden, db_hidden, dw_output, db_output

# Update parameters
def update_parameters(dw_hidden, db_hidden, dw_output, db_output, learning_rate, clip_value):
    global weights_input_to_hidden, biases_hidden, weights_hidden_to_output, biases_output

    # Apply gradient clipping
    dw_hidden = np.clip(dw_hidden, -clip_value, clip_value)
    db_hidden = np.clip(db_hidden, -clip_value, clip_value)
    dw_output = np.clip(dw_output, -clip_value, clip_value)
    db_output = np.clip(db_output, -clip_value, clip_value)

    weights_input_to_hidden -= learning_rate * dw_hidden
    biases_hidden -= learning_rate * db_hidden
    weights_hidden_to_output -= learning_rate * dw_output
    biases_output -= learning_rate * db_output

# Generate synthetic data
def generate_data(num_samples):
    X = np.random.uniform(-1, 1, (num_samples, 2))
    Y = (1 - X[:, 0])**2 + 100 * (X[:, 1] - X[:, 0]**2)**2
    Y = Y.reshape(-1, 1)  # Reshape Y to be a column vector
    return X, Y

# Training function
def train(X, Y, epochs, learning_rate, clip_value, print_interval):
    for epoch in range(epochs):
        # Forward propagation
        z, yhat = forward_propagation(X)

        # Compute loss
        loss = np.mean((yhat - Y) ** 2)  # Mean squared error

        # Backpropagation
        dw_hidden, db_hidden, dw_output, db_output = backpropagation(X, Y, z, yhat)

        # Update parameters
        update_parameters(dw_hidden, db_hidden, dw_output, db_output, learning_rate, clip_value)

        # Print loss every few epochs
        if epoch % print_interval == 0:
            print(f"Loss at epoch {epoch}: {loss}")

        # Check for NaN values
        if np.isnan(loss) or np.isinf(loss):
            print("Stopping training due to numerical instability.")
            break

# Initialize network parameters with Xavier initialization
input_size = 2          # Number of input features
hidden_size = 10        # Number of neurons in the hidden layer
output_size = 1         # Number of neurons in the output layer
np.random.seed(0)
weights_input_to_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
weights_hidden_to_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
biases_hidden = np.zeros((1, hidden_size))
biases_output = np.zeros((1, output_size))

# Hyperparameters
epochs = 30000
learning_rate = 0.0001  # Reduced learning rate
clip_value = 1.0        # Gradient clipping threshold
print_interval = 5000   # Interval for printing the loss

# Generate training and testing data
X_train, Y_train = generate_data(200)
X_test, Y_test = generate_data(100)

# Train the model
train(X_train, Y_train, epochs, learning_rate, clip_value, print_interval)

# Test the model
_, yhat_train = forward_propagation(X_train)
_, yhat_test = forward_propagation(X_test)
train_error = np.mean((yhat_train - Y_train) ** 2)
test_error = np.mean((yhat_test - Y_test) ** 2)
print(f"Training Error: {train_error}")
print(f"Test Error: {test_error}")

# Plot training data and predicted values
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.ravel(), cmap='viridis', label="True Values")
plt.scatter(X_train[:, 0], X_train[:, 1], c=yhat_train.ravel(), cmap='coolwarm', marker="x", label="Predicted Values")
plt.title("Training Data: True vs. Predicted")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.colorbar()
plt.show()
