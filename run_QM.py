
import time
import csv
import psi4
import numpy as np


start_time = time.time()

# Set the output file
psi4.set_output_file("output_2912")


# function to create coulomb matrix
def extract_atomic_data(molecule):
    n_atoms = molecule.natom()
    atomic_data = []
    for i in range(n_atoms):
        symbol = molecule.symbol(i).strip('0123456789')  # Remove numeric suffixes
        charge = atomic_number_lookup.get(symbol)
        if charge is None:
            raise ValueError(f"Unknown atomic symbol: {symbol}")
        coords = np.array([molecule.x(i), molecule.y(i), molecule.z(i)])  # Explicitly extract coordinates
        atomic_data.append((charge, coords))
    return atomic_data


def compute_coulomb_matrix(atomic_data):
    n_atoms = len(atomic_data)
    coulomb_matrix = np.zeros((n_atoms, n_atoms))
    for i, (Zi, pos_i) in enumerate(atomic_data):
        for j, (Zj, pos_j) in enumerate(atomic_data):
            if i == j:
                # Diagonal term
                coulomb_matrix[i, j] = 0.5 * Zi ** 2.4
            else:
                # Off-diagonal terms
                distance = np.linalg.norm(pos_i - pos_j)  # NumPy array subtraction
                coulomb_matrix[i, j] = Zi * Zj / distance
    return coulomb_matrix


# lookup table

atomic_number_lookup = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
}


# function to calculate the hydrogen bond length
def calculate_distance_between_atoms(molecule, atom1_label, atom2_label):
    """
    Calculates the distance between two atoms in a Psi4 molecule.

    Parameters:
        molecule (psi4.core.Molecule): Psi4 molecule object.
        atom1_label (str): Label of the first atom (e.g., 'H6').
        atom2_label (str): Label of the second atom (e.g., 'O2').

    Returns:
        float: Distance between the two atoms, or None if an error occurs.
    """
    n_atoms = molecule.natom()  # Number of atoms in the molecule
    atom_labels = [molecule.label(i) for i in range(n_atoms)]  # Get all atom labels

    try:
        # Find the indices of the specified atom labels
        atom1_idx = atom_labels.index(atom1_label)
        atom2_idx = atom_labels.index(atom2_label)
    except ValueError as e:
        print(f"Error: {e}. Ensure atom labels {atom1_label} and {atom2_label} are correct.")
        return None

    # Get atomic coordinates
    coord1 = np.array([molecule.x(atom1_idx), molecule.y(atom1_idx), molecule.z(atom1_idx)])
    coord2 = np.array([molecule.x(atom2_idx), molecule.y(atom2_idx), molecule.z(atom2_idx)])

    # Calculate distance
    distance = np.linalg.norm(coord1 - coord2)

    return distance


# determine the hydrogen bond angle (make sure these are the right atoms)

def calculate_bond_angle(molecule, atom1_label, atom2_label, atom3_label):

    n_atoms = molecule.natom()
    atom_labels = [molecule.label(i) for i in range(n_atoms)]

    try:
        idx1 = atom_labels.index(atom1_label)
        idx2 = atom_labels.index(atom2_label)
        idx3 = atom_labels.index(atom3_label)
    except ValueError as e:
        print(f"Error: {e}. Ensure atom labels {atom1_label}, {atom2_label}, and {atom3_label} are correct.")
        return None

    # Get atomic coordinates
    coord1 = np.array([molecule.x(idx1), molecule.y(idx1), molecule.z(idx1)])
    coord2 = np.array([molecule.x(idx2), molecule.y(idx2), molecule.z(idx2)])
    coord3 = np.array([molecule.x(idx3), molecule.y(idx3), molecule.z(idx3)])

    # Calculate vectors and angle
    vector1 = coord1 - coord2
    vector2 = coord3 - coord2
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees




#geometries.py is the ~500 geometries to loop through 

from geometries import * 

# collect the geometries into a list
geometries1 = [eval(f"geometry_{i}") for i in range(1, 501)]


#there is an issue with geometry 335 as it does not converge so it needs replacing or skipping 
geometries1 = [eval(f"geometry_{i}") for i in range(1, 501) if i != 335]




# extract geometries into numpy arrays
def psi4_to_numpy(geometry):
    """converts psi4 geometry to numpy array"""
    natoms = geometry.natom()
    coords = np.empty((natoms, 4), dtype=object)

    for i in range(natoms):
        coords[i, 0] = geometry.symbol(i)
        coords[i, 1] = geometry.x(i)
        coords[i, 2] = geometry.y(i)
        coords[i, 3] = geometry.z(i)
    return coords

# run the extract calculation function for all geometries
geometries_numpy = [psi4_to_numpy(geometry) for geometry in geometries1]


#concatenate the geometries
all_geometries = np.concatenate((geometries_numpy), axis=0)

#print the dimensions of all_geometries
print(all_geometries.shape)



#set psi4 settings 
psi4.set_memory('20 GB')
psi4.set_num_threads(12)
psi4.set_options({
    "basis": "cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-6,
    "d_convergence": 1e-6

})

# lists to hold data
energies = []
frequencies = []
dipoles = []
coulomb_matrices = []
ir_intensities = []
reduced_masses = []
distances = []
angles = []
hydrogen_bond_length = []
hydrogen_bond_angle = []

#loop through geometries 
for i, geom in enumerate(geometries1, start=1):
    psi4.core.clean()
    molecule = geom  # Set a new geometry for each run

    try:
        # Perform frequency calculations
        energy, wfn = psi4.frequencies('mp2', molecule=molecule, return_wfn=True)
        vib_info = wfn.frequency_analysis

        # Append calculated properties
        energies.append(energy)
        ir_intensities.append(vib_info['IR_intensity'].data)
        frequencies.append(vib_info["omega"].data)
        reduced_masses.append(vib_info['mu'].data)

        # Calculate dipole moment
        dipole = np.array(wfn.variable("CURRENT DIPOLE"))
        dipole_magnitude = np.linalg.norm(dipole)
        dipoles.append(dipole_magnitude)

        # Extract atomic data and compute the Coulomb matrix
        atomic_data = extract_atomic_data(geom)
        coulomb_matrix = compute_coulomb_matrix(atomic_data)
        coulomb_matrices.append(coulomb_matrix)

        # Append hydrogen bond length and angle
        distance = calculate_distance_between_atoms(geom, 'H7', 'O1')
        if distance is not None:
            distances.append(distance)

        angle = calculate_bond_angle(geom, 'O1', 'O2', 'H7')
        if angle is not None:
            angles.append(angle)


        # Save the frequencies as a CSV
        with open('freq_all2912.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write all frequency rows for all geometries
            for freq_list in frequencies:
                writer.writerow(freq_list)

    except psi4.SCFConvergenceError as scf_err:
        print(f"SCF did not converge for geometry {i}: {scf_err}")
        continue  # Skip to the next geometry

    except Exception as e:
        print(f"Failed geometry {i}: {e}")
        continue  # Skip to the next geometry


# Convert lists to NumPy arrays for ML training
energies = np.array(energies)
# frequencies = np.array(frequencies)
dipoles = np.array(dipoles)
coulomb_matrices = np.array(coulomb_matrices)
ir_intensities = np.array(ir_intensities)
reduced_masses = np.array(reduced_masses)
distances = np.array(distances)
angles = np.array(angles)

print(f"angles shape: {np.array(angles).shape}")
print(f"distances shape: {np.array(distances).shape}")

# Verify data dimensions
print(f"Energies shape: {np.array(energies).shape}")
#print(f"energies shape: {np.array(frequencies).shape}")

print("dipole and coulomb matrix size")
print(f"Dipoles shape: {np.array(dipoles).shape}")

print(f"Coulomb matrices shape: {np.array(coulomb_matrices).shape}")

# Save arrays to files (optional)
np.save('energies_2912.npy', energies)
# np.save('frequencies.npy', frequencies)
np.save('dipoles_2912.npy', dipoles)
np.save('coulomb_matrices_2912.npy', coulomb_matrices)
np.save('ir_intensities_2912.npy', ir_intensities)
np.save('reduced_masses_all2912.npy', reduced_masses)
np.save('angles_all2912.npy', angles)
np.save('distances_all2912.npy', distances)
np.save('geometries2912.npy', all_geometries)



print("job complete")




