#with mass without damping
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Transmissibility function
def transmissibility(omega, k, c, m, Y):
    num = np.sqrt(k**2 + (c * omega)**2)
    denom = np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
    return (num / denom) * Y  # Multiply by base displacement Y

# Example dataset: omega (frequencies in rad/s) and transmissibility ratio |X| / |Y|
omega_data = np.array([28.53714286,31.42857143,33.37714286,38.09142857,38.28,
                       39.72571429,42.42857143,42.99428571,44.25142857,45.50857143,
                       48.33714286,55.56571429,63.48571429,66.62857143,77.31428571])  # Replace with your actual omega data

transmissibility_data = np.array([0.155905512,0.173228346,0.196850394,0.217322835,0.245669291,
                                    0.349606299,0.881889764,1.606299213,1.700787402,1.480314961,
                                    0.522834646,0.321259843,0.187401575,0.151181102,0.119685039])  # Replace with your actual transmissibility data

# Set constants
Y = 6.35  # Amplitude of base displacement in mm
initial_k = 2161.359501  # Initial guess for spring constant k (N/m)
c = 0  # No damping case
m = 1.1  # Mass with extra mass attached (kg)

# Curve fitting to find k
popt, _ = curve_fit(lambda omega, k: transmissibility(omega, k, c, m, Y), omega_data, transmissibility_data, p0=[initial_k])

# Extracting the optimal spring constant k
optimal_k = popt[0]

# Plotting the fitted curve and data
omega_fine = np.linspace(min(omega_data), max(omega_data), 500)
fitted_transmissibility = transmissibility(omega_fine, optimal_k, c, m, Y)

plt.plot(omega_data, transmissibility_data, 'bo', label='Experimental data')
plt.plot(omega_fine, fitted_transmissibility, 'r-', label=f'Fitted curve (k = {optimal_k:.2f} N/m)')
plt.xlabel('Frequency (omega)')
plt.ylabel('Transmissibility (|X| / |Y|)')
plt.legend()
plt.title('Transmissibility vs Frequency (No Damping, Extra Mass)')
plt.show()

print(f"Optimal spring constant k: {optimal_k:.2f} N/m")
