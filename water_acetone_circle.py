#create multiple geometries where acetone is bonded at 2.71 angstroms to water around a circle

#use autode to make H3 the origin
#https://duartegroup.github.io/autodE/examples/molecules.html

import autode as ade

# Define the molecule
water = ade.Molecule(name='h2o', smiles='O')
print("before translation", water.coordinates)

h_atom = water.atoms[2]
water.translate(vec=-h_atom.coord)
print("after translation", water.coordinates)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Atom labels
labels = ['O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

# Define the initial coordinates of the acetone molecule
acetone_coordinates = np.array([
    [0.0000000000, 0.0000000000, 2.7190000000],  # Oxygen
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

# Setup the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through 360 degrees, rotating around the Y-axis every 72 degrees
for i in range(0, 360, 72):
    theta = np.radians(i)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotated_coordinates = np.dot(acetone_coordinates, rotation_matrix.T)

    # Plot the rotated points
    ax.scatter(rotated_coordinates[:, 0], rotated_coordinates[:, 1], rotated_coordinates[:, 2], label=f'{i}°')

    # Print out the updated coordinates with labels
    print(f"Rotation at {i} degrees:")
    for label, coord in zip(labels, rotated_coordinates):
        print(f"{label} {coord}")

# Setting labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(title="Rotation Angle")
ax.set_title('Independent Rotations of Acetone around the Y-axis')

# Show the plot
plt.show()