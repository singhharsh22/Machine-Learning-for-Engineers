#without mass with damping
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Transmissibility function
def transmissibility(omega, c, k, m, Y):
    num = np.sqrt(k**2 + (c * omega)**2)
    denom = np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
    return (num / denom) * Y  # Multiply by base displacement Y

# Example dataset: omega (frequencies in rad/s) and transmissibility ratio |X| / |Y|
omega_data = np.array([27.09142857,32.05714286,40.79428571,45.82285714,48.33714286,
                       52.36,55.56571429,56.06857143,57.64,58.70857143,
                       61.6,67.88571429,69.77142857,72.91428571,75.42857143])  # Replace with your actual omega data

transmissibility_data = np.array([0.116535433,0.138582677,0.193700787,0.283464567,0.31023622,
                                  0.57007874,0.951181102,0.963779528,0.881889764,0.793700787,
                                  0.538582677,0.280314961,0.234645669,0.204724409,0.176377953])  # Replace with your actual transmissibility data

# Set constants
Y = 6.35  # Amplitude of base displacement in mm
fixed_k = 2215.98  # Fixed spring constant k (N/m)
initial_c = 7.377635581  # Initial guess for damping coefficient c (Ns/m)
m = 0  # No mass attached (kg) in this case

# Curve fitting to find the optimal damping coefficient c
popt, _ = curve_fit(lambda omega, c: transmissibility(omega, c, fixed_k, m, Y), omega_data, transmissibility_data, p0=[initial_c])

# Extracting the optimal damping coefficient c
optimal_c = popt[0]

# Plotting the fitted curve and data
omega_fine = np.linspace(min(omega_data), max(omega_data), 500)
fitted_transmissibility = transmissibility(omega_fine, optimal_c, fixed_k, m, Y)

plt.plot(omega_data, transmissibility_data, 'bo', label='Experimental data')
plt.plot(omega_fine, fitted_transmissibility, 'r-', label=f'Fitted curve (k = {fixed_k:.2f} N/m, c = {optimal_c:.2f} Ns/m)')
plt.xlabel('Frequency (omega)')
plt.ylabel('Transmissibility (|X| / |Y|)')
plt.legend()
plt.title('Transmissibility vs Frequency (No Mass, With Damping)')
plt.show()

print(f"Optimal damping coefficient c: {optimal_c:.2f} Ns/m")
