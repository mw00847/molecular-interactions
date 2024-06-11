"""original cartesian coordinates
 C    1.240000    1.879200    0.019500
 C   -0.150200    1.784300    0.029300
 C    2.010100    0.716800   -0.020400
 C   -0.786000    0.535700    0.000800
 C    1.388800   -0.533600   -0.052600
 C   -0.000000   -0.628800   -0.041000
 N   -2.200000    0.521200    0.013600
 C   -3.053500   -0.560200    0.007000
 C   -4.532900   -0.194800    0.011500
 H   -4.733100    0.877200    0.108800
 H   -5.019700   -0.723100    0.836500
 H   -4.986900   -0.554200   -0.917900
 H    2.009500   -1.424100   -0.094200
 H   -0.486100   -1.593700   -0.066200
 H    1.715600    2.857800    0.044700
 O    3.390200    0.745600   -0.023700
 O   -2.684700   -1.726300   -0.008800
 H   -0.743600    2.695900    0.060400
 H    3.696700    1.664600   -0.058500
 H   -2.634000    1.433300    0.029400
 O    4.439500   -1.986800   -0.035100
 H    4.293300   -1.031400   -0.148000
 H    4.431200   -2.102700    0.926800

"""

import psi4
import numpy as np
import matplotlib.pyplot as plt

psi4.core.clean()

# Set memory & output
psi4.set_memory('10 GB')




# distances is a list between 1 and 3 with steps of 0.1

distances = np.arange(1, 2.5, 0.1)


# Output file prefix to print out the dimer geometries
output_file_prefix = 'dimer_W'  

# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))

for i in range(len(distances)):
    dimerW = psi4.geometry("""
    0 1
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
    
    B1    =    1.39347
    B2    =    1.39493
    A2    =  119.61878
    B3    =    1.40145
    A3    =  120.87334
    D3    =    0.05771
    B4    =    1.39662
    A4    =  120.07456
    D4    =    0.10930
    B5    =    1.39211
    A5    =  120.32048
    D5    =  359.80312
    B6    =    1.41413
    A6    =  117.55388
    D6    =  179.97316
    B7    =    1.37765
    A7    =  128.86434
    D7    =  180.81458
    B8    =    1.52386
    A8    =  114.40721
    D8    =  180.57694
    B9    =    1.09487
    A9    =  114.36561
    D9    =  354.85656
    B10   =     1.09394
    A10   =   108.56861
    D10   =   232.99406
    B11   =     1.09502
    A11   =   108.74155
    D11   =   116.57538
    B12   =     1.08627
    A12   =   118.72700
    D12   =   179.35033
    B13   =     1.08072
    A13   =   120.63622
    D13   =   179.97402
    B14   =     1.08834
    A14   =   119.80456
    D14   =   180.10363
    B15   =     1.38040
    A15   =   122.31903
    D15   =   180.40673
    B16   =     1.22313
    A16   =   124.16767
    D16   =     0.17907
    B17   =     1.08817
    A17   =   119.15603
    D17   =   180.00581
    B18   =     0.96939
    A18   =   109.63176
    D18   =   356.01159
    B19   =     1.01021
    A19   =   114.86335
    D19   =     0.83771                       


    --
    H   16 B23 13 A21 5 D21
    O   21 B21 5 A20 3 D20
    H   22 B22 13 A22 5 D22
    
    
    
    
    
    B20   =     2.49500
    A20   =   137.77358
    D20   =     4.14079
    B21   =     0.97309
    A21   =    68.25310
    D21   =   348.60070
    B22   =     0.96889
    A22   =    92.41581
    D22   =    92.04279
    B23 = """ + str(distances[i]) + """
    
    
    """)


    # Print the XYZ geometry for increasing distance
    xyz_geometry = dimerW.to_string(dtype='xyz')
    print(f"XYZ Geometry for Distance {distances[i]}:\n{xyz_geometry}\n")

    # Save the XYZ geometry to an output file
    output_file_name = f"{output_file_prefix}{distances[i]:.2f}.xyz"
    with open(output_file_name, 'w') as output_file:
        output_file.write(xyz_geometry)


    psi4.set_options({'scf_type': 'df',
                      'freeze_core': 'true',
                      })

    # sapt calculation
    psi4.energy('sapt0/jun-cc-pvdz', molecule=dimerW)

    eelst[i] = psi4.variable('SAPT ELST ENERGY')*psi4.constants.hartree2kcalmol
    eexch[i] = psi4.variable('SAPT EXCH ENERGY')*psi4.constants.hartree2kcalmol
    eind[i] = psi4.variable('SAPT IND ENERGY')*psi4.constants.hartree2kcalmol
    edisp[i] = psi4.variable('SAPT DISP ENERGY')*psi4.constants.hartree2kcalmol
    esapt[i] = psi4.variable('SAPT TOTAL ENERGY')*psi4.constants.hartree2kcalmol

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

#save the plot

plt.savefig('dimerW_sapt.png')


#save the data as csv

data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('dimerW_sapt.csv', data, delimiter=',', header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')