#fit a curve to the water-acetone distance data

import numpy as np

# Load the data from text files
distances = np.loadtxt('water_acetone_distance.txt')
energies = np.loadtxt('water_acetone_energy.txt')


import matplotlib.pyplot as plt
#plot scatter plot
plt.scatter(distances, energies)
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy vs Distance')
plt.show()



