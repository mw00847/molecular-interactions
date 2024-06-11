#compare the SAPT of two water dimers with the dummy atom on either water molecule

import psi4
import numpy as np
import matplotlib.pyplot as plt

#clear the memory
psi4.core.clean()

# Set memory & output
psi4.set_memory('5 GB')

# Calculate at distances between 1-3 angstroms with a step size of 0.1
distances = np.arange(2.5, 9, 0.5)

# Output file prefix to print out each geometry at increasing distance
output_file_prefix = 'water_geometry_'

# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))

# Loop over distances
for i in range(len(distances)):
    # Define water_dimer
    water_dimer = psi4.geometry("""
    O1
    H1 O1 0.96
    H2 O1 0.96 H1 104.5
    --
    O2 O1 {} H1 5.0 H2 0.0
    X O2 1.0 O1 120.0 H2 180.0
    H3 O2 0.96 X 52.25 O1 90.0
    H4 O2 0.96 X 52.25 O1 -90.0
    units angstrom
    symmetry c1
    """.format(distances[i]))

    # Print the XYZ geometry for the current water dimer
    xyz_geometry = water_dimer.to_string(dtype='xyz')
    print(f"XYZ Geometry for Distance {distances[i]}:\n{xyz_geometry}\n")

    # Save the XYZ geometry to an output file
    output_file_name = f"{output_file_prefix}{distances[i]:.2f}.xyz"
    with open(output_file_name, 'w') as output_file:
        output_file.write(xyz_geometry)

    # Set Psi4 options
    psi4.set_options({'scf_type': 'df', 'freeze_core': 'true'})

    # SAPT calculation for water_dimer
    psi4.energy('sapt0/jun-cc-pvdz', molecule=water_dimer)

    # Store energies
    eelst[i] = psi4.variable('SAPT ELST ENERGY') * psi4.constants.hartree2kcalmol
    eexch[i] = psi4.variable('SAPT EXCH ENERGY') * psi4.constants.hartree2kcalmol
    eind[i] = psi4.variable('SAPT IND ENERGY') * psi4.constants.hartree2kcalmol
    edisp[i] = psi4.variable('SAPT DISP ENERGY') * psi4.constants.hartree2kcalmol
    esapt[i] = psi4.variable('SAPT TOTAL ENERGY') * psi4.constants.hartree2kcalmol

    # Clean up Psi4 core
    psi4.core.clean()

# Plotting for water_dimer
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("water dimer SAPT energies with increasing distance")
plt.plot(distances, eelst, label='Electrostatics')
plt.plot(distances, eexch, label='Exchange')
plt.plot(distances, eind, label='Induction')
plt.plot(distances, edisp, label='Dispersion')
plt.plot(distances, esapt, label='SAPT Total')
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (kcal/mol)')
plt.legend()

# Define arrays for water_dimer2
eelst2 = np.zeros((len(distances)))
eexch2 = np.zeros((len(distances)))
eind2 = np.zeros((len(distances)))
edisp2 = np.zeros((len(distances)))
esapt2 = np.zeros((len(distances)))

# Loop over distances again for water_dimer2
for i in range(len(distances)):
    # Define water_dimer2
    water_dimer2 = psi4.geometry("""
    O1
    H2 1 1.0
    H3 1 1.0 2 104.52
    x4 2 1.0 1 90.0 3 180.0
    --
    O5 2 {} 4 90.0 1 180.0
    H6 5 1.0 2 120.0 4 90.0
    H7 5 1.0 2 120.0 4 -90.0
    """.format(distances[i]))

    # SAPT calculation for water_dimer2
    #psi4.energy('sapt0/jun-cc-pvdz', molecule=water_dimer2)

    # Store energies for water_dimer2
    #eelst2[i] = psi4.variable('SAPT ELST ENERGY') * psi4.constants.hartree2kcalmol
    #eexch2[i] = psi4.variable('SAPT EXCH ENERGY') * psi4.constants.hartree2kcalmol
    #eind2[i] = psi4.variable('SAPT IND ENERGY') * psi4.constants.hartree2kcalmol
    #edisp2[i] = psi4.variable('SAPT DISP ENERGY') * psi4.constants.hartree2kcalmol
    #esapt2[i] = psi4.variable('SAPT TOTAL ENERGY') * psi4.constants.hartree2kcalmol


    #calculate the energy at each distance
    psi4.set_options({'scf_type': 'df', 'freeze_core': 'true'})



    # Clean up Psi4 core
    psi4.core.clean()

# Plotting for water_dimer2
plt.subplot(1, 2, 2)
plt.title("water dimer2 SAPT energies with increasing distance")
plt.plot(distances, eelst2, label='Electrostatics')
plt.plot(distances, eexch2, label='Exchange')
plt.plot(distances, eind2, label='Induction')
plt.plot(distances, edisp2, label='Dispersion')
plt.plot(distances, esapt2, label='SAPT Total')
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (kcal/mol)')
plt.legend()

# Save the plot
plt.savefig('water.png')

# Save the data as csv for water_dimer
data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('water.csv', data, delimiter=',',
           header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')

# Save the data as csv for water_dimer2
data2 = np.column_stack((distances, eelst2, eexch2, eind2, edisp2, esapt2))
np.savetxt('water2.csv', data2, delimiter=',',
           header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')

plt.show()
