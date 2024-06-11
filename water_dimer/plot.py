import matplotlib.pyplot as plt
import numpy as np

# Read the numpy files from the directory
energy_values1 = np.load('energy_values_dimer1.npy')
energy_values2 = np.load('energy_values_dimer2.npy')
energy_values3 = np.load('energy_values_dimer3.npy')
energy_values4 = np.load('energy_values_dimer4.npy')
energy_values5 = np.load('energy_values_dimer5.npy')
energy_values6 = np.load('energy_values_dimer6.npy')
energy_values7 = np.load('energy_values_dimer7.npy')
energy_values9 = np.load('energy_values_dimer9.npy')

r_values1 = np.load('r_values_dimer1.npy')
r_values2 = np.load('r_values_dimer2.npy')
r_values3 = np.load('r_values_dimer3.npy')
r_values4 = np.load('r_values_dimer4.npy')
r_values5 = np.load('r_values_dimer5.npy')
r_values6 = np.load('r_values_dimer6.npy')
r_values7 = np.load('r_values_dimer7.npy')
r_values9 = np.load('r_values_dimer9.npy')

# List of energy arrays, r_values, and corresponding labels
data = [
    (energy_values1, r_values1, 'Dimer 1'),
    (energy_values2, r_values2, 'Dimer 2'),
    (energy_values3, r_values3, 'Dimer 3'),
    (energy_values4, r_values4, 'Dimer 4'),
    (energy_values5, r_values5, 'Dimer 5'),
    (energy_values6, r_values6, 'Dimer 6'),
    (energy_values7, r_values7, 'Dimer 7'),
    (energy_values9, r_values9, 'Dimer 9')
]

# Create a single plot
plt.figure(figsize=(12, 8))

# Plot each energy array on the same plot
for energy_values, r_values, label in data:
    plt.plot(r_values, energy_values, 'o-', label=label)

# Set labels and title
plt.xlabel('Distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy Profile of Water Dimers')
plt.legend()

# Show the plot
plt.show()
