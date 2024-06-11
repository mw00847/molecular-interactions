import psi4
import numpy as np
import matplotlib.pyplot as plt

# Define the water molecule geometry
water = """
O
H 1 0.96
H 1 0.96 2 104.5
"""

# Define the acetone molecule geometry
acetone = """
  C     0.0000      0.0000      0.0000
  C     1.5200      0.0000      0.0000
  C    -1.5200      0.0000      0.0000
  O     0.0000      1.2000      0.0000
  H     1.8800     -0.9200      0.0000
  H     1.8800      0.9200      0.0000
  H    -1.8800     -0.9200      0.0000
  H    -1.8800      0.9200      0.0000
"""

# Set Psi4 options
psi4.set_memory('4 GB')
psi4.set_num_threads(4)

# Define the basis set
basis_set = 'cc-pVDZ'


# Define a function to calculate interaction energy and save geometries
def interaction_energy(distance):
    # Define the geometry of water and acetone molecules with increasing distance
    mol_geometry = f"""
    O    0.000000   0.000000   0.000000
    H    0.000000   0.757160   0.586260
    H    0.000000  -0.757160   0.586260
    --
    C    0.0000      0.0000     {distance}
    C    1.5200      0.0000     {distance}
    C   -1.5200      0.0000     {distance}
    O    0.0000      1.2000     {distance}
    H    1.8800     -0.9200     {distance}
    H    1.8800      0.9200     {distance}
    H   -1.8800     -0.9200     {distance}
    H   -1.8800      0.9200     {distance}
    """

    # Save the geometry to a file
    filename = f"geometry_{distance:.2f}.xyz"
    with open(filename, "w") as f:
        f.write(mol_geometry)

    # Create the molecule
    mol = psi4.geometry(mol_geometry)

    # Calculate the interaction energy
    psi4.set_options({'basis': basis_set})
    e_total = psi4.energy('scf', molecule=mol)
    e_water = psi4.energy('scf', molecule=psi4.geometry(water))
    e_acetone = psi4.energy('scf', molecule=psi4.geometry(acetone))
    e_int = e_total - (e_water + e_acetone)
    return e_int


# Define the range of distances
distances = np.linspace(2, 10.0, 50)  # from 2.0 to 10.0 Å
energies = []

# Calculate interaction energies for each distance and save geometries
for d in distances:
    e_int = interaction_energy(d)
    energies.append(e_int)
    print(f"Distance: {d:.2f} Å, Interaction Energy: {e_int:.6f} Hartree")

# Plot the Lennard-Jones curve
plt.plot(distances, energies, '-o')
plt.xlabel('Distance (Å)')
plt.ylabel('Interaction Energy (Hartree)')
plt.title('Lennard-Jones Potential Curve for Water and Acetone Molecules')
plt.grid(True)
plt.show()
