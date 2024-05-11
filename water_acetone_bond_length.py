import psi4
import numpy as np
import matplotlib.pyplot as plt

distances = np.arange(0.2, 3, 0.2)






# psi4 initialization
psi4.set_memory('4GB')
psi4.core.clean_options()

# Define basis and method
basis = '6-31G'
method = 'B3LYP'

# Array to store energies
energies = []
distance=[]

#define the distance and calculate the energy
for i in range(len(distances)):
    complex =psi4.geometry("""

    O  
    H   1 B1
    H   1 B2 2 A2
    O   3 B3 1 A3 2 D3
    C   4 B4 3 A4 1 D4
    C   5 B5 4 A5 3 D5
    C   5 B6 4 A6 3 D6
    H   6 B7 5 A7 4 D7
    H   6 B8 5 A8 4 D8
    H   6 B9 5 A9 4 D9
    H   7 B10 5 A10 4 D10
    H   7 B11 5 A11 4 D11
    H   7 B12 5 A12 4 D12



    B1   =     0.98784
    B2   =     0.99021
    A2   =   113.17081
    B3   =     """ + str(distances[i]) + """
    A3   =    90.00000
    D3   =   270.00000
    B4   =     1.51706
    A4   =   149.61206
    D4   =   181.35901
    B5   =     1.03200
    A5   =   120.00000
    D5   =   151.47490
    B6   =     1.03200
    A6   =   120.00000
    D6   =   331.47489
    B7   =     1.07000
    A7   =   109.47100
    D7   =   360.00000
    B8   =     1.07000
    A8   =   109.47128
    D8   =   240.00003
    B9   =     1.07000
    A9   =   109.47135
    D9   =   120.00010
    B10  =      1.07000
    A10  =    109.47100
    D10  =      0.00000
    B11  =      1.07000
    A11  =    109.47128
    D11  =   240.00003
    B12  =      1.07000
    A12  =    109.47135
    D12  =   120.00010

    """)

    # Compute energy
    psi4.energy(f"{method}/{basis}", molecule=complex)

    energy = psi4.energy(f"{method}/{basis}", molecule=complex)
    energies.append(energy)
    distance.append(distances[i])
    print(f"Distance at {distances[i]} Angstrom: Energy = {energy:.10f}")

    psi4.core.clean()

# Convert energies to a NumPy array
energy_array = np.array(energies)
distances = np.array(distance)


#plot
plt.plot(distance,energy_array)
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy vs Distance')
plt.show()
print(energy_array)
print(distances)

# Save the data to a file
np.savetxt('energy_vs_distance.dat', np.column_stack((distances, energy_array)), header='Distance (Angstrom)  Energy (Hartree)', fmt='%12.8f')



