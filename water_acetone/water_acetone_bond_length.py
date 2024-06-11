import psi4
import numpy as np
import matplotlib.pyplot as plt

# psi4 initialization
psi4.set_memory('4GB')
psi4.core.clean_options()

# Define basis and method
basis = 'cc-pVDZ'
method = 'MP2'

# Define the geometries of water and acetone molecules
water_geometry = """
O
H 1 0.96
H 1 0.96 2 104.5
"""

acetone_geometry = """
  C     0.0000      0.0000      0.0000
  C     1.5200      0.0000      0.0000
  C    -1.5200      0.0000      0.0000
  O     0.0000      1.2000      0.0000
  H     1.8800     -0.9200      0.0000
  H     1.8800      0.9200      0.0000
  H    -1.8800     -0.9200      0.0000
  H    -1.8800      0.9200      0.0000
"""

# Define distances
distances = np.arange(2.5, 10, 0.2)

# Array to store energies
energies = []

#define the distance and calculate the energy
define interaction_energy(distances):
    for i in range(len(distances)):
        complex_molecule =psi4.geometry("""
        O  
        C   1 B1
        C   2 B2 1 A2
        C   2 B3 1 A3 3 D3
        H   3 B4 2 A4 1 D4
        H   3 B5 2 A5 1 D5
        H   3 B6 2 A6 1 D6
        H   4 B7 2 A7 1 D7
        H   4 B8 2 A8 1 D8
        H   4 B9 2 A9 1 D9
        
        O   5 B10 3 A10 2 D10
        H   11 B11 5 A11 3 D11
        H   11 B12 5 A12 3 D12
        
        B1    =    1.24936
        B2    =    1.51013
        A2    =  121.66586
        B3    =    1.51075
        A3    =  120.35514
        D3    =  180.23108
        B4    =    1.09269
        A4    =  110.15654
        D4    =    5.65995
        B5    =    1.09706
        A5    =  110.53506
        D5    =  127.82955
        B6    =    1.09826
        A6    =  109.78985
        D6    =  245.32774
        B7    =    1.09122
        A7    =  110.15709
        D7    =    2.15041
        B8    =    1.09743
        A8    =  110.26921
        D8    =  241.27396
        B9    =    1.09709
        A9    =  110.44641
        D9    =  123.34113
        B10   =     2.33940
        A10   =   140.58089
        D10   =   350.28619
        B11   =     0.98673
        A11   =    70.41125
        D11   =     3.08675
        B12   =     0.97461
        A12   =   121.96400
        D12   =   103.58000


    
        """)




        # Compute total energy
        e_total = psi4.energy(f"{method}/{basis}", molecule=complex_molecule)

        # Compute energies of individual molecules
        water_molecule = psi4.geometry(water_geometry)
        acetone_molecule = psi4.geometry(acetone_geometry)

        e_water = psi4.energy(f"{method}/{basis}", molecule=water_molecule)
        e_acetone = psi4.energy(f"{method}/{basis}", molecule=acetone_molecule)

        # Compute interaction energy
        e_int = e_total - (e_water + e_acetone)


# Calculate interaction energies for each distance
for d in distances:
    e_int = interaction_energy(d)
    energies.append(e_int)
    print(f"Distance: {d:.2f} Å, Interaction Energy: {e_int:.6f} Hartree")

# Plot the interaction energy curve
plt.plot(distances, energies, '-o')
plt.xlabel('Distance (Å)')
plt.ylabel('Interaction Energy (Hartree)')
plt.title('Interaction Energy Curve for Water and Acetone Molecules')
plt.grid(True)
plt.show()