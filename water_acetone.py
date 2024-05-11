#this script calculates the vibrational frequencies of
#water acetone at increasing hydrogen bond distances

import numpy as np
import psi4
import numpy
import matplotlib.pyplot as plt

psi4.core.clean()

psi4.set_memory('4 GB')

# set the basis set
basis_set = 'cc-pVDZ'
method = 'scf/' + basis_set

#distance to increase the hydrogen bond
distances=np.arange(1,3.5,0.5)

#define array for vibration
vibrations=np.zeros((len(distances)))
ir_intensities=np.zeros((len(distances)))


#declare the geometry for increasing distances
for i in range(len(distances)):
    complex =psi4.geometry("""
O
H   1 B1
H   1 B2 2 A2
--
O   2 B3 1 A3 3 D3
C   4 B4 2 A4 1 D4
H   5 B5 4 A5 2 D5
H   5 B6 4 A6 2 D6

B1    =    0.99025
B2    =    0.99025
A2    =  104.51212
B3    =    """ + str(distances[i]) + """
A3    = 109.77648
D3    =   29.18723
B4    =    1.21946
A4    =   88.63850
D4    =   74.74498
B5    =    1.08440
A5    =  120.00405
D5    =  315.91391
B6    =    1.08438
A6    =  119.99850
D6    =  135.91167
""")

    # run the frequency calculation
    psi4.frequency(method, molecule=complex, return_c1=True)
    frequencies = frequencies().to_array()
    intensities = ('IR_INTENSITIES')

    #store the frequencies and intensities
    vibrations[i] = frequencies[0]
    ir_intensities[i] = intensities[0]


#plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(distances, vibrations, marker='o')
plt.title('Vibrational Frequencies vs Distance')
plt.xlabel('Distance (Å)')
plt.ylabel('Frequency (cm^-1)')

plt.subplot(1, 2, 2)
plt.plot(distances, ir_intensities, marker='o', color='red')
plt.title('IR Intensities vs Distance')
plt.xlabel('Distance (Å)')
plt.ylabel('Intensity (km/mol)')

plt.tight_layout()
plt.show()

print(vibrations)
print(ir_intensities)



