import psi4
import numpy as np
import matplotlib.pyplot as plt

psi4.core.clean()

# Set memory & output
psi4.set_memory('10 GB')

# distances is a list between 1 and 3 with steps of 0.1

distances = np.arange(1, 2.5, 0.1)

# Output file prefix to print out the dimer geometries
output_file_prefix = 'dimer_Y'  

# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))

for i in range(len(distances)):
    dimerY = psi4.geometry("""
    
    C  
    C   1 B1
    C   1 B2 2 A2
    C   2 B3 1 A3 3 D3
    C   3 B4 1 A4 2 D4
    C   5 B5 3 A5 1 D5
    N   4 B6 2 A6 1 D6
    C   7 B7 4 A7 2 D7
    C   8 B8 7 A8 4 D8
    H   9 B9 8 A9 7 D9
    H   9 B10 8 A10 7 D10
    H   9 B11 8 A11 7 D11
    H   5 B12 3 A12 1 D12
    H   6 B13 5 A13 3 D13
    H   1 B14 2 A14 3 D14
    O   3 B15 1 A15 2 D15
    O   8 B16 7 A16 4 D16
    H   2 B17 1 A17 3 D17
    H   16 B18 3 A18 1 D18
    H   7 B19 4 A19 2 D19
    
    --
    
    H   16 B23 15 A22 1 D22
    O   21 B22 1 A20 2 D20 
    H   22 B21 15 A21 1 D21
    
    
    
    
    B1   =     1.38972
    B2   =     1.39716
    A2   =   119.21694
    B3   =     1.40455
    A3   =   121.28469
    D3   =   359.93660
    B4   =     1.39425
    A4   =   120.09899
    D4   =     0.06829
    B5   =     1.39560
    A5   =   120.68375
    D5   =   359.98904
    B6   =     1.41411
    A6   =   117.53118
    D6   =   180.10375
    B7   =     1.37662
    A7   =   128.78927
    D7   =   180.34012
    B8   =     1.52356
    A8   =   114.50643
    D8   =   180.29779
    B9   =     1.09466
    A9   =   114.38091
    D9   =   358.18063
    B10  =      1.09426
    A10  =    108.62319
    D10  =    236.34939
    B11  =      1.09469
    A11  =    108.68318
    D11  =    119.94397
    B12  =      1.08876
    A12  =    120.16163
    D12  =    180.03661
    B13  =      1.08061
    A13  =    120.85988
    D13  =    179.99296
    B14  =      1.08595
    A14  =    121.67285
    D14  =    179.57994
    B15  =      1.38058
    A15  =    117.44180
    D15  =    180.29719
    B16  =      1.22392
    A16  =    124.09683
    D16  =      0.16703
    B17  =      1.08843
    A17  =    118.97445
    D17  =    180.13619
    B18  =      0.96953
    A18  =    109.46465
    D18  =    183.83727
    B19  =      1.01030
    A19  =    114.81686
    D19  =      0.51358
    B20  =      2.46629
    A20  =    137.75564
    D20  =    177.65827
    B21  =      0.96898
    A21  =     96.00935
    D21  =    269.20824
    B22  =      0.97337
    A22  =     69.19351
    D22  =    11.17327
    
    B23 = """ + str(distances[i]) + """
        
    """)

    # Print the XYZ geometry for increasing distance
    xyz_geometry = dimerY.to_string(dtype='xyz')
    print(f"XYZ Geometry for Distance {distances[i]}:\n{xyz_geometry}\n")

    # Save the XYZ geometry to an output file
    output_file_name = f"{output_file_prefix}{distances[i]:.2f}.xyz"
    with open(output_file_name, 'w') as output_file:
        output_file.write(xyz_geometry)

    psi4.set_options({'scf_type': 'df',
                      'freeze_core': 'true',
                      })
    # sapt calculation

    psi4.energy('sapt0/jun-cc-pvdz', molecule=dimerY)

    eelst[i] = psi4.variable('SAPT ELST ENERGY')
    eexch[i] = psi4.variable('SAPT EXCH ENERGY')
    eind[i] = psi4.variable('SAPT IND ENERGY')
    edisp[i] = psi4.variable('SAPT DISP ENERGY')
    esapt[i] = psi4.variable('SAPT TOTAL ENERGY')

    psi4.core.clean()

# plotting

plt.plot(distances, eelst, label='Electrostatics')
plt.plot(distances, eexch, label='Exchange')
plt.plot(distances, eind, label='Induction')
plt.plot(distances, edisp, label='Dispersion')
plt.plot(distances, esapt, label='SAPT Total')
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Energy (kcal/mol)')
plt.legend()
#plt.show()

# save the plot

plt.savefig('dimerY_sapt.png')

# save the data as csv

data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('dimerY_sapt.csv', data, delimiter=',',
           header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')