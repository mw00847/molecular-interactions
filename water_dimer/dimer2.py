import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#above is needed to avoid the below error
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

import autode as ade
import numpy as np
import matplotlib.pyplot as plt
import psi4
import glob

#'optimised' water dimer
dimer = ade.Molecule(atoms=[


        ade.Atom("O", -1.54242900, -0.09751400, -0.02750100),
        ade.Atom("O", 1.39404800, 0.00116400, -0.07467400),
        ade.Atom("H", -0.58610800, 0.04744900, -0.07109200),
        ade.Atom("H", -1.91349200, 0.72347000, 0.30431400),
        ade.Atom("H", 1.71046800, -0.73825200, 0.45296400),
        ade.Atom("H", 1.97618200, 0.73812600, 0.13121700)
    ])

# Indices of atoms to be translated
idxs = (0,2,3)  # NOTE: atoms are 0 indexed

# Calculate the unit vector
vec = dimer.atoms.nvector(1, 2)
unit_vec = vec / np.linalg.norm(vec)

# Initial distance
r0 = dimer.atoms.distance(1, 2)

#range of distances
r_values = np.arange(1, 5, 0.2)

# Function to update coordinates
def update_coordinates(dimer, r, idxs, unit_vec, r0):
    new_dimer = dimer.copy()  # Create a copy to avoid modifying the original dimer
    for i in idxs:
        atom = new_dimer.atoms[i]
        atom.coord += unit_vec * (r - r0)
    return new_dimer

#loop through increasing distances between
for r in r_values:
    translated_dimer = update_coordinates(dimer, r, idxs, unit_vec, r0)
    filename = f"dimer2_translated_{r:.1f}.xyz"
    translated_dimer.print_xyz_file(filename=filename)

# Clear psi4 output
psi4.core.clean()


def read_xyz(filename):
    with open(filename, 'r') as file:
        contents = file.read()
    return contents


import glob

#create a list of the XYZ files
xyz_files = glob.glob("dimer2_translated_*.xyz")
print(xyz_files)

def run_psi4(xyz_files):

    def read_xyz(filename):
        with open(filename, 'r') as file:
            contents = file.read()
        return contents

    # Set up Psi4 use b3lyp and 6-31G
    psi4.set_options({'basis': '6-31G**', 'reference': 'rhf'})
    psi4.set_memory('4 GB')

    # Loop through the XYZ files and calculate energy
    energies = []
    for xyz_file in xyz_files:
        # Read the XYZ file contents
        xyz_contents = read_xyz(xyz_file)

        # Create Psi4 molecule from XYZ contents
        molecule = psi4.geometry(xyz_contents)

        # Calculate the energy
        energy = psi4.energy('scf/sto-3g', molecule=molecule)
        energies.append((float(xyz_file.split('_')[-1].split('.xyz')[0]), energy))
        print(f"Energy for {xyz_file}: {energy} Hartree")

    # Sort energies based on r values
    energies.sort(key=lambda x: x[0])

    # Unpack r_values and energies for plotting
    r_values, energy_values = zip(*energies)

    # Plot the energy profile
    plt.plot(r_values, energy_values, 'o-')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Energy (Hartree)')
    plt.title('Energy Profile of Water Dimer Translation')
    plt.show()

    #save the data as a numpy array
    np.save('energy_values_dimer2.npy', energy_values)
    np.save('r_values_dimer2.npy', r_values)

# Call the run_psi4 function
run_psi4(xyz_files)
