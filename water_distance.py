
#calculate the non covalent energies with increasing hydrogen bond distance 


#http://bbs.keinsci.com/thread-15902-1-1.html
#https://github.com/Psi4Education/psi4education/blob/master/labs/Symmetry_Adapted_Perturbation_Theory/sapt0_student.ipynb


import psi4
import numpy as np
import matplotlib.pyplot as plt

psi4.core.clean()

# Set memory & output
psi4.set_memory('4 GB')

# calculate at distances of between 1-3 angstroms with a step size of 0.1 
distances = np.arange(1, 3, 0.1)

# Output file prefix to print out each geometry at increasing distance
output_file_prefix = 'water_geometry_'  


# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))


#define each of the geometries using distances and starting geometry
for i in range(len(distances)):
     #Define water di
    water_dimer = psi4.geometry("""
    

    O1
    H2 1 1.0
    H3 1 1.0 2 104.52
    x4 2 1.0 1 90.0 3 180.0
    --
    O5 2 """ + str(distances[i]) + """ 4 90.0 1 180.0
    H6 5 1.0 2 120.0 4 90.0
    H7 5 1.0 2 120.0 4 -90.0
    
    
    """)
# Print the XYZ geometry for the current water dimer
    xyz_geometry = water_dimer.to_string(dtype='xyz')
    print(f"XYZ Geometry for Distance {distances[i]}:\n{xyz_geometry}\n")

    # Save the XYZ geometry to an output file
    output_file_name = f"{output_file_prefix}{distances[i]:.2f}.xyz"
    with open(output_file_name, 'w') as output_file:
        output_file.write(xyz_geometry)


    
    psi4.set_options({'scf_type': 'df',
                      'freeze_core': 'true',
                      })

    # sapt calculation
    psi4.energy('sapt0/jun-cc-pvdz', molecule=water_dimer)


    #energies
    eelst[i] = psi4.variable('SAPT ELST ENERGY')*psi4.constants.hartree2kcalmol
    eexch[i] = psi4.variable('SAPT EXCH ENERGY')*psi4.constants.hartree2kcalmol
    eind[i] = psi4.variable('SAPT IND ENERGY')*psi4.constants.hartree2kcalmol
    edisp[i] = psi4.variable('SAPT DISP ENERGY')*psi4.constants.hartree2kcalmol
    esapt[i] = psi4.variable('SAPT TOTAL ENERGY')*psi4.constants.hartree2kcalmol

    psi4.core.clean()

    # plotting
plt.title("water dimer SAPT energies with increasing distance")
plt.plot(distances, eelst, label='Electrostatics')
plt.plot(distances, eexch, label='Exchange')
plt.plot(distances, eind, label='Induction')
plt.plot(distances, edisp, label='Dispersion')
plt.plot(distances, esapt, label='SAPT Total')
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (kcal/mol)')
plt.legend()


# save the plot

plt.savefig('water.png')

# save the data as csv
data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('water.csv', data, delimiter=',',
            header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')
plt.show()