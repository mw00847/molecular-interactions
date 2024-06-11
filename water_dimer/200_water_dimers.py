import numpy as np
import matplotlib.pyplot as plt
import psi4

# Define the coordinates of the two water molecules
water1 = np.array([
    [-2.3771464565, -0.3288779707, 0.0000000000],
    [-1.4071464565, -0.3288779707, 0.0000000000],
    [-2.7004762682, -0.6092567668, 0.8704858205]
])

# Extract the positions of O1, H2, and H3 for the first water molecule
O1 = water1[0]
H2 = water1[1]
H3 = water1[2]

# Define the bond lengths and bond angle for a water molecule
bond_length_OH = 0.96
bond_angle_HOH = 104.5

# Define the number of new coordinates to generate
num_new_coordinates = 200

# Initialize an empty list to store the new positions of O4, H5, and H6
new_water2_positions = []

# Calculate the center of the circle as the midpoint between O1 and H2
center = (O1 + H2) / 2.0

# Generate random angles for the circular motion
random_angles = np.linspace(0, 2 * np.pi, num_new_coordinates, endpoint=False)

# Perform the circular motion and generate new coordinates while preserving bond lengths and angles
for angle in random_angles:
    # Calculate the new positions while preserving bond lengths and angles
    new_O4 = center + 2.0 * np.array([np.cos(angle), np.sin(angle), 0])
    new_H5 = new_O4 + bond_length_OH * np.array(
        [np.cos(np.radians(bond_angle_HOH)), np.sin(np.radians(bond_angle_HOH)), 0])
    new_H6 = new_O4 + bond_length_OH * np.array(
        [np.cos(np.radians(-bond_angle_HOH)), np.sin(np.radians(-bond_angle_HOH)), 0])

    new_water2_positions.append([new_O4, new_H5, new_H6])

# Convert the positions to a numpy array
new_water2_positions = np.array(new_water2_positions)

# Define a list to store molecule objects
molecules = []

# Generate molecule objects for the 200 water dimers
for i, positions in enumerate(new_water2_positions):
    mol_str = f"""
    0 1
    O1 {positions[0, 0]:.6f} {positions[0, 1]:.6f} {positions[0, 2]:.6f}
    H2 {positions[1, 0]:.6f} {positions[1, 1]:.6f} {positions[1, 2]:.6f}
    H3 {positions[2, 0]:.6f} {positions[2, 1]:.6f} {positions[2, 2]:.6f}
    O4 {O1[0]:.6f} {O1[1]:.6f} {O1[2]:.6f}
    H5 {H2[0]:.6f} {H2[1]:.6f} {H2[2]:.6f}
    H6 {H3[0]:.6f} {H3[1]:.6f} {H3[2]:.6f}
    """
    molecule = psi4.geometry(mol_str)
    molecules.append(molecule)

# Create a 3D scatter plot of the positions of all molecules
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the positions of all 200 water dimers, including the starting water molecule (water1)
for i, molecule in enumerate(molecules):
    ax.scatter(molecule.x(0), molecule.y(0), molecule.z(0), label=f'Molecule {i+1}', marker='o', s=30, alpha=0.7)

# Plot the positions of the O, H, and H atoms for the starting water1 molecule
ax.scatter(O1[0], O1[1], O1[2], label='O1 (water1)', marker='o', s=100, c='red')
ax.scatter(H2[0], H2[1], H2[2], label='H2 (water1)', marker='x', s=50, c='blue')
ax.scatter(H3[0], H3[1], H3[2], label='H3 (water1)', marker='^', s=50, c='green')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot title and legend
ax.set_title('Positions of Water Dimers (Including water1)')


# Show the plot
plt.show()



# Generate molecule objects for the 200 water dimers and print their coordinates
for i, positions in enumerate(new_water2_positions):
    mol_str = f"""
    0 1
    O1 {positions[0, 0]:.6f} {positions[0, 1]:.6f} {positions[0, 2]:.6f}
    H2 {positions[1, 0]:.6f} {positions[1, 1]:.6f} {positions[1, 2]:.6f}
    H3 {positions[2, 0]:.6f} {positions[2, 1]:.6f} {positions[2, 2]:.6f}
    O4 {O1[0]:.6f} {O1[1]:.6f} {O1[2]:.6f}
    H5 {H2[0]:.6f} {H2[1]:.6f} {H2[2]:.6f}
    H6 {H3[0]:.6f} {H3[1]:.6f} {H3[2]:.6f}
    """
    molecule = psi4.geometry(mol_str)
    molecules.append(molecule)
    

    with open(r"C:\Users\Mat\PycharmProjects\psi4\water_dimer_coordinates.txt", "w") as f:
        for i, positions in enumerate(new_water2_positions):
            f.write(f"O1 {positions[0, 0]:.6f} {positions[0, 1]:.6f} {positions[0, 2]:.6f}\n")
            f.write(f"H2 {positions[1, 0]:.6f} {positions[1, 1]:.6f} {positions[1, 2]:.6f}\n")
            f.write(f"H3 {positions[2, 0]:.6f} {positions[2, 1]:.6f} {positions[2, 2]:.6f}\n")
            f.write(f"O4 {O1[0]:.6f} {O1[1]:.6f} {O1[2]:.6f}\n")
            f.write(f"H5 {H2[0]:.6f} {H2[1]:.6f} {H2[2]:.6f}\n")
            f.write(f"H6 {H3[0]:.6f} {H3[1]:.6f} {H3[2]:.6f}\n\n")
    


    print(f"O1 {positions[0, 0]:.6f} {positions[0, 1]:.6f} {positions[0, 2]:.6f}")
    print(f"H2 {positions[1, 0]:.6f} {positions[1, 1]:.6f} {positions[1, 2]:.6f}")
    print(f"H3 {positions[2, 0]:.6f} {positions[2, 1]:.6f} {positions[2, 2]:.6f}")
    print(f"O4 {O1[0]:.6f} {O1[1]:.6f} {O1[2]:.6f}")
    print(f"H5 {H2[0]:.6f} {H2[1]:.6f} {H2[2]:.6f}")
    print(f"H6 {H3[0]:.6f} {H3[1]:.6f} {H3[2]:.6f}")
    print()


