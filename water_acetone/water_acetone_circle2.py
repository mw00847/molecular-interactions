import autode as ade
import psi4
import numpy as np
import matplotlib.pyplot as plt

# Psi4 setup
psi4.core.set_output_file('psi4_output.dat', False)
psi4.set_memory('2 GB')
psi4.set_num_threads(4)

theorylevel = 'b3lyp/def2-SVP'

import autode as ade

# Define the molecule
water = ade.Molecule(name='h2o', smiles='O')
print("before translation", water.coordinates)

h_atom = water.atoms[2]
water.translate(vec=-h_atom.coord)
print("after translation", water.coordinates)

water=psi4.geometry("""
    0 1
    O -8.2720e-01  5.4430e-01 -0.0000e+00
    H -1.6511e+00 -7.0000e-04 -0.0000e+00
    H 0.0000e+00  0.0000e+00  0.0000e+00
    symmetry c1
    """)


# Define the initial coordinates of the acetone molecule
acetone = ade.Molecule(smiles='CC(=O)C')
acetone.translate(vec=np.array([0.0, 0.0, 2.719]))  # Place the oxygen 2.719 Å away from the origin


# Define function to rotate around the y-axis
def rotate_y(molecule, theta):
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    new_coords = np.dot(molecule.coordinates, rotation_matrix)
    #molecule.coordinates = new_coords

    print(new_coords)

    # Energy scan
    angles = np.linspace(0, 360, 2)  # Degrees
    energies = []

    for angle in angles:
        theta = np.radians(angle)
        rotate_y(acetone, theta)

    # Combine the molecules for Psi4 calculation
    combined_geometry=psi4.geometry(new_coords+water.coordinates)


    combined = ade.Molecule(atoms=water.atoms + acetone.atoms)
    combined.single_point(method='HF/6-31G*')

    psi4.energy(theorylevel, molecule=combined)

    # Extract energy
    energies.append(combined.energy)

    # Reset acetone position before next rotation
    acetone.translate(vec=np.array([0.0, 0.0, 2.719]) - acetone.atoms[0].coord.to_array())

    # Plotting the energy vs. angle
    plt.plot(angles, energies, '-o')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Energy (Hartrees)')
    plt.title('Potential Energy Surface of Acetone-Water Interaction')
    plt.grid(True)
    plt.show()

