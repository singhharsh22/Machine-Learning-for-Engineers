import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Input data (Propellant age and temperature, and Pass/Fail for each sample)
age = np.array([15.5, 23.75, 8, 17, 5.5, 19, 24, 2.5, 7.5, 11])
temp = np.array([40, 23.25, 17, 21, 10, 12, 20, 12, 15, 26])
pass_fail = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)

# Combine age and temperature into a single matrix of features (X) and add a bias term (1s column)
X = np.column_stack((np.ones(age.shape[0]), age, temp))  # Adding intercept term (bias)
y = pass_fail  # Target variable (0 or 1 for fail or pass)
m = len(y)  # Number of training examples

# Sigmoid function to map the linear combination of features to a probability between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function for logistic regression (cross-entropy loss)
def cost_function(w, epsilon=1e-10):
    z = np.dot(X, w)  # Linear combination of input features and weights
    h = sigmoid(z)    # Logistic (sigmoid) function to get predicted probabilities
    h = np.clip(h, epsilon, 1 - epsilon)  # Clip h to avoid log(0) or log(1) issues
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # Cross-entropy loss
    return cost

# Gradient of the cost function (used for gradient descent)
def gradient_function(w):
    z = np.dot(X, w)  # Linear combination of input features and weights
    h = sigmoid(z)    # Logistic (sigmoid) function to get predicted probabilities
    gradient = (1/m) * np.dot(X.T, h - y)  # Compute gradient vector
    return gradient

# Line search to adaptively find the optimal step size for gradient descent
def line_search(cost_function, gradient_function, w):
    beta = 0.1  # Armijo condition constant
    stepsize = 1.0  # Initial step size
    trial = 100  # Number of trials to find optimal step size
    tau = 0.5  # Step size reduction factor
    grad = gradient_function(w)  # Compute the gradient at the current weights
    for _ in range(trial):
        new_w = w - stepsize * grad  # Update weights using gradient descent with current step size
        # Check if the Armijo condition is satisfied
        if cost_function(new_w) <= cost_function(w) - beta * stepsize * np.dot(grad, grad):
            break  # Exit the loop if Armijo condition is satisfied
        stepsize *= tau  # Reduce the step size if condition is not satisfied
    return stepsize

# Gradient descent with line search to minimize the cost function
def gradient_descent(max_iter=100000, epsilon=1e-3):
    w = np.zeros(X.shape[1])  # Initialize weights to zeros (for bias, age, and temperature)
    for i in range(max_iter):
        grad = gradient_function(w)  # Compute gradient of the cost function
        norm_grad = np.linalg.norm(grad)  # Compute the norm of the gradient (for convergence check)
        if norm_grad < epsilon:  # Stop if gradient is smaller than the tolerance (epsilon)
            break
        stepsize = line_search(cost_function, gradient_function, w)  # Find optimal step size
        w -= stepsize * grad  # Update weights using gradient descent step
        if i % 1000 == 0:  # Print progress every 1000 iterations
            print(f"Iteration {i}, Cost: {cost_function(w)}, Gradient Norm: {norm_grad}")
    return w, cost_function(w), i  # Return the final weights, cost, and number of iterations

# Run the gradient descent optimization algorithm
optimal_w, min_cost, iterations = gradient_descent()

# Output the results
print(f"Optimal weights: {optimal_w}")  # Optimal weights for logistic regression
print(f"Minimum cost: {min_cost}")  # Minimum value of the cost function
print(f"Number of iterations: {iterations}")  # Number of iterations until convergence

# Plotting the scatter plot and probability contour for passing/failing
age_plot, temp_plot = np.meshgrid(np.linspace(0, 30, 200), np.linspace(0, 50, 200))  # Generate grid for age and temperature
z_plot = optimal_w[0] + optimal_w[1] * age_plot + optimal_w[2] * temp_plot  # Linear combination for the grid
p_plot = sigmoid(z_plot)  # Sigmoid function to get probabilities for the grid

# Define a vibrant colormap from vibrant red (fail) to vibrant green (pass)
cmap = LinearSegmentedColormap.from_list('pass_fail', ['red', 'limegreen'], N=256)

# Plotting the contour and scatter plot with 10 regions for probability
plt.contourf(age_plot, temp_plot, p_plot, levels=np.linspace(0, 1, 11), cmap=cmap, alpha=0.8)  # 10 regions from 0 to 1
colorbar = plt.colorbar(label='Probability of Pass')  # Color bar indicating probability levels

# Set colorbar ticks to intervals of 0.1
colorbar.set_ticks(np.arange(0, 1.1, 0.1))

# Scatter plot with vibrant fill colors and black edges
plt.scatter(age[pass_fail == 1], temp[pass_fail == 1], color='limegreen', edgecolor='black', label='Pass')  # Vibrant green fill for passing samples
plt.scatter(age[pass_fail == 0], temp[pass_fail == 0], color='red', edgecolor='black', label='Fail')  # Vibrant red fill for failing samples

plt.xlabel('Propellant Age (Weeks)')  # X-axis label
plt.ylabel('Storage Temperature (°C)')  # Y-axis label
plt.legend()  # Show legend
plt.title('Passing Probability Contour with Logistic Regression')  # Plot title
plt.show()  # Display the plot
