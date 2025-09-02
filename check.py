import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.314  # J/mol·K
T_range = np.arange(1000, 3500, 500)  # Temperature range in Kelvin

# Thermodynamic data (assumed values, replace with actual data)
delta_H_f = {'N2': 0, 'O2': 0, 'NO': 90250, 'NO2': 33180}  # J/mol
S0 = {'N2': 191.5, 'O2': 205.0, 'NO': 210.7, 'NO2': 240.0}  # J/mol·K

def gibbs_free_energy(T, species):
    return delta_H_f[species] - T * S0[species]

# Equilibrium constant calculation
def equilibrium_constant(T, delta_G0):
    return np.exp(-delta_G0 / (R * T))

# Initialize results
moles_N2 = []
moles_O2 = []
moles_NO = []
moles_NO2 = []

for T in T_range:
    # Gibbs free energy changes
    delta_G0_1 = gibbs_free_energy(T, 'NO') * 2 - gibbs_free_energy(T, 'N2') - gibbs_free_energy(T, 'O2')
    delta_G0_2 = gibbs_free_energy(T, 'NO2') - gibbs_free_energy(T, 'NO') - 0.5 * gibbs_free_energy(T, 'O2')
    
    # Equilibrium constants
    Kp1 = equilibrium_constant(T, delta_G0_1)
    Kp2 = equilibrium_constant(T, delta_G0_2)
    
    # Initial guesses
    x = 0.1  # Initial guess for NO
    y = 0.01  # Initial guess for NO2
    
    # Iterative solution to find equilibrium moles
    for _ in range(1000):  # Iteration loop
        f1 = x ** 2 / ((1 - x/2) * (1 - x/2)) - Kp1  # Reaction 1
        f2 = y / ((x - y) * np.sqrt(1 - x/2 - y/2)) - Kp2  # Reaction 2
        df1_dx = 2 * x / ((1 - x/2) ** 2)  # Derivative w.r.t x
        df2_dy = 1 / ((x - y) * np.sqrt(1 - x/2 - y/2))  # Derivative w.r.t y
        
        # Newton-Raphson update
        x -= f1 / df1_dx
        y -= f2 / df2_dy
    
    moles_N2.append(1 - x/2)
    moles_O2.append(1 - x/2 - y/2)
    moles_NO.append(x - y)
    moles_NO2.append(y)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(T_range, moles_N2, label='N2')
plt.plot(T_range, moles_O2, label='O2')
plt.plot(T_range, moles_NO, label='NO')
plt.plot(T_range, moles_NO2, label='NO2')
plt.xlabel('Temperature (K)')
plt.ylabel('Moles')
plt.title('Equilibrium Moles of Species vs. Temperature')
plt.legend()
plt.grid(True)
plt.show()