import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Line search function
def line_search(objective_function, gradient, x):
    beta = 0.1
    stepsize = 1
    trial = 100
    tau = 0.5
    for i in range(trial):
        fx1 = objective_function(x)
        fx2 = objective_function(x - stepsize * gradient)
        c = -beta * stepsize * np.dot(gradient, gradient)
        if fx2 - fx1 <= c:
            break
        else:
            stepsize = tau * stepsize
    return stepsize

# Objective function and gradient
objective_function = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[1]**2 + x[0] - 7)**2
gradient_function = lambda x: np.array([
    4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[1]**2 + x[0] - 7),
    2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[1]**2 + x[0] - 7)
])

# Optimization parameters
maxit = 10000
epsilon = 1.e-6
x_store = np.zeros((2, maxit))
x = np.array([0.0, 0.0])

# Steepest descent method
for i in range(maxit):
    x_store[0, i] = x[0]
    x_store[1, i] = x[1]
    gradient = gradient_function(x)
    b = np.linalg.norm(gradient)
    if b < epsilon:
        break
    stepsize = line_search(objective_function, gradient, x)
    x = x - stepsize * gradient

# Final results
minimum_value = objective_function(x)
print("Minimum value:", minimum_value)
print("Minimum location:", x)
print("Iteration:", i)

# Plotting the results
X, Y = np.meshgrid(np.linspace(-6, 6, 400), np.linspace(-6, 6, 400))
Z = np.array([objective_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

fig = plt.figure(figsize=(20, 10))

# First subplot: 3D Surface Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.contour3D(X, Y, Z, 60, cmap='viridis')
ax1.plot(x_store[0, :i+1], x_store[1, :i+1], [objective_function([x, y]) for x, y in zip(x_store[0, :i+1], x_store[1, :i+1])], color='red', linewidth=3)
ax1.set_xlabel('$x_{0}$')
ax1.set_ylabel('$x_{1}$')
ax1.set_zlabel('$f(x)$')
ax1.set_title('3D Surface Plot with Convergence Path')
ax1.view_init(20, 20)

# Second subplot: 3D Surface Plot with different view
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.contour3D(X, Y, Z, 60, cmap='viridis')
ax2.plot(x_store[0, :i+1], x_store[1, :i+1], [objective_function([x, y]) for x, y in zip(x_store[0, :i+1], x_store[1, :i+1])], color='red', linewidth=3)
ax2.set_xlabel('$x_{0}$')
ax2.set_ylabel('$x_{1}$')
ax2.set_zlabel('$f(x)$')
ax2.set_title('3D Surface Plot with Convergence Path (Different View)')
ax2.axes.zaxis.set_ticklabels([])
ax2.view_init(90, -90)

plt.show()
