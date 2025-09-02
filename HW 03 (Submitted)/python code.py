import numpy as np
import matplotlib.pyplot as plt

# Function to minimize
def objective_function(var):
    # The objective function f(x1, x2) = (x1^2 + x2 - 11)^2 + (x2^2 + x1 - 7)^2
    return (var[0]**2 + var[1] - 11)**2 + (var[1]**2 + var[0] - 7)**2

# Gradient of the function
def gradient(var):
    # Partial derivatives of the objective function with respect to x1 and x2
    grad_var1 = 4*var[0]*(var[0]**2 + var[1] - 11) + 2*(var[1]**2 + var[0] - 7)
    grad_var2 = 2*(var[0]**2 + var[1] - 11) + 4*var[1]*(var[1]**2 + var[0] - 7)
    # Return the gradient as a vector [df/dx1, df/dx2]
    return np.array([grad_var1, grad_var2])

# Line search function to find the optimal step size
def find_step_size(current_point, descent_direction):
    # Initial step size
    step_size = 1.0
    # Reduction factor for step size (used to gradually decrease step size)
    reduction_factor = 0.5
    # Small constant to ensure sufficient decrease in function value (Wolfe condition)
    c1 = 1e-4
    # Perform backtracking line search to find optimal step size
    while objective_function(current_point + step_size * descent_direction) > objective_function(current_point) + c1 * step_size * np.dot(gradient(current_point), descent_direction):
        # Reduce step size if the condition is not met
        step_size *= reduction_factor
    # Return the optimal step size
    return step_size

# Steepest descent method
def steepest_descent_method(initial_point, tolerance=1e-3):
    # Start at the initial point
    current_point = initial_point
    # List to store the path of points visited during optimization
    path = [current_point]
    # Iterate until the norm of the gradient is smaller than the tolerance
    while np.linalg.norm(gradient(current_point)) > tolerance:
        # Compute the descent direction as the negative gradient
        descent_direction = -gradient(current_point)
        # Find the optimal step size using line search
        step_size = find_step_size(current_point, descent_direction)
        # Update the current point by moving in the descent direction
        current_point = current_point + step_size * descent_direction
        # Store the new point in the path
        path.append(current_point)
    # Return the path of all points visited
    return np.array(path)

# Initial point for the steepest descent method
initial_point = np.array([0.0, 0.0])
# Run the steepest descent method and get the path of points
path = steepest_descent_method(initial_point)

# Print the coordinates of the point where the function is minimized
print(f"argmin(f(x)) = [{', '.join(map(str, path[-1]))}]")
#print(f"argmin(f(x)) = [{', '.join(f'{x:.5f}' for x in path[-1])}]")
# The above line prints the coordinates of the last point in the path array, which is the point where the function is minimized.

# Plotting the surface and contour plots with the convergence path


# Generate values for var1 and var2 (corresponding to x1 and x2) for the plot
var1_vals = np.linspace(-6, 6, 400)
var2_vals = np.linspace(-6, 6, 400)
Var1, Var2 = np.meshgrid(var1_vals, var2_vals)
Objective = objective_function([Var1, Var2])

# Create a figure with two subplots
fig = plt.figure(figsize=(14, 6))

# Surface plot of the objective function
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Var1, Var2, Objective, cmap='viridis', alpha=0.7)
# Add the convergence path to the surface plot
path = np.array(path)
ax1.plot(path[:, 0], path[:, 1], objective_function(path.T), 'ro-', linewidth=2, markersize=5)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1,x2)')
ax1.set_title('Surface Plot of f(x1,x2) with Convergence Path')

# Contour plot of the objective function
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(Var1, Var2, Objective, levels=np.logspace(0, 5, 35), cmap='viridis')
# Plot the path of convergence on the contour plot
ax2.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=5)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot with Convergence Path')

# Show the plots
plt.show()