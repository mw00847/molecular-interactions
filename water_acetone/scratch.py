import psi4
import numpy as np
import matplotlib.pyplot as plt

# Define the initial coordinates of the acetone molecule
#reference H bond length = 2.719, 1.93

acetone_coordinates = np.array([
    [0.0000000000, 0.0000000000, 1.7],  # Oxygen
    [0.6509000509, -0.4065066944, 4.0276504673],  # Carbon
    [0.5238430544, 0.1760398279, 4.8699806098],   # Carbon
    [1.2207390549, -1.2655839220, 4.0755443769],  # Carbon
    [-0.0915662883, 1.0182786779, 4.6316424872],  # Hydrogen
    [1.4775363292, 0.5173389026, 5.2147763160],   # Hydrogen
    [0.0538245234, -0.4035001659, 5.6368678088],  # Hydrogen
    [1.2394900833, -1.7351916764, 3.1142863700],  # Hydrogen
    [0.7949164979, -1.9365541655, 4.7920456555],  # Hydrogen
    [2.2186302201, -1.0157162938, 4.3699580158]   # Hydrogen
])
labels = ["O", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

# Define the water molecule geometry
water_geometry = """
0 1
O -8.2720e-01  5.4430e-01 -0.0000e+00
H -1.6511e+00 -7.0000e-04 -0.0000e+00
H 0.0000e+00  0.0000e+00 0.0000e+00
symmetry c1
"""

# psi4 initialization
psi4.set_memory('4GB')
psi4.core.clean_options()

# Define basis and method
basis = '6-31G'
method = 'B3LYP'

# Array to store energies
energies = []
angles = []
frequencies=[]

# Define rotations and calculate energies
for i in range(0, 360, 72):
    theta = np.radians(i)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotated_coordinates = np.dot(acetone_coordinates, rotation_matrix.T)

    # Create the Psi4 geometry string combining acetone and water
    acetone_str = "\n".join(f"{label} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}" for label, coord in zip(labels, rotated_coordinates))
    full_geometry = psi4.geometry(f"""
    {acetone_str}
    {water_geometry}
    """)

    # Compute energy
    energy = psi4.energy(f"{method}/{basis}", molecule=full_geometry)
    energies.append(energy)
    angles.append(i)
    print(f"Rotation at {i} degrees: Energy = {energy:.10f}")

    # Compute frequencies
    vib_info = psi4.frequency(f"{method}/{basis}", molecule=full_geometry, return_wfn=True)[1]
    frequencies.append(vib_info.frequencies().to_array())



# Convert energies to a NumPy array
energy_array = np.array(energies)

#convert frequencies to a numpy array
frequency_array=np.array(frequencies)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(angles, energy_array, 'o-')
#plt.xlim(0, 360)
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy Profile of Rotated Acetone with Water Molecule')
plt.grid(True)
plt.show()

#plotting frequencies
plt.figure(figsize=(10, 6))
plt.plot(angles, frequency_array, 'o-')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Frequency (cm^-1)')
plt.title('Frequency Profile of Rotated Acetone with Water Molecule')
plt.show()
print(frequency_array)