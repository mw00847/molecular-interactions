import psi4
import numpy as np
import matplotlib.pyplot as plt


# Set memory & output
psi4.set_memory('10 GB')

# distances is a list between 1 and 3 with steps of 0.1

distances = np.arange(1, 2.5, 0.1)

# Output file prefix to print out the dimer geometries
output_file_prefix = 'dimer_V'  


# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))

for i in range(len(distances)):
    dimerV = psi4.geometry("""
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

    B1  =      1.39284
    B2  =      1.40151
    A2  =    119.97404
    B3  =      1.40124
    A3  =    121.13127
    D3  =    359.99506
    B4  =      1.40065
    A4  =    118.91670
    D4  =      0.00485
    B5  =      1.39188
    A5  =    121.21342
    D5  =    359.99501
    B6  =      1.41727
    A6  =    117.57705
    D6  =    180.00294
    B7  =      1.37405
    A7  =    129.05020
    D7  =    180.00055
    B8  =      1.52509
    A8  =    114.43500
    D8  =    179.99101
    B9  =      1.09499
    A9  =    114.42535
    D9  =      0.01815
    B10 =       1.09450
    A10 =     108.67085
    D10 =     238.21776
    B11 =       1.09461
    A11 =     108.67080
    D11 =     121.81778
    B12 =       1.08580
    A12 =     118.54586
    D12 =     180.00228
    B13 =       1.08094
    A13 =     120.84444
    D13 =     179.99694
    B14 =       1.08688
    A14 =     120.10831
    D14 =     180.00144
    B15 =       1.36112
    A15 =     122.93072
    D15 =     180.00283
    B16 =       1.22460
    A16 =     124.38736
    D16 =     359.99005
    B17 =       1.08911
    A17 =     119.09772
    D17 =     180.00318
    B18 =       0.98180
    A18 =     109.74013
    D18 =       0.00900
    B19 =       1.01013
    A19 =     114.66736
    D19 =     359.99515
    --
    0 1


    O   19 B20 16 A20 3 D20
    H   21 B21 19 A21 16 D21
    H   21 B22 19 A22 16 D22

    B20 =      """ + str(distances[i]) + """
    A20 =     173.38664
    D20 =     181.83837
    B21 =       0.97021
    A21 =     107.38328
    D21 =     302.24948
    B22 =       0.97022
    A22 =     107.13206
    D22 =      54.53430

    """)


    # Print the XYZ geometry for increasing distance
    xyz_geometry = dimerV.to_string(dtype='xyz')
    print(f"XYZ Geometry for Distance {distances[i]}:\n{xyz_geometry}\n")

    # Save the XYZ geometry to an output file
    output_file_name = f"{output_file_prefix}{distances[i]:.2f}.xyz"
    with open(output_file_name, 'w') as output_file:
        output_file.write(xyz_geometry)

    # calculation

    psi4.set_options({'scf_type': 'df',
                  'freeze_core': 'true'})

    #use sapt1 for just electrostatics and exchange

    #psi4.energy('sapt1/jun-cc-pvdz', molecule=dimerV)
    psi4.energy('sapt0/jun-cc-pvdz', molecule=dimerV)



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

#make the y axis start at 0 and end at 5

#plt.ylim(-0.5,1.8)


#plt.show()



#save the plot

plt.savefig('dimerV_sapt.png')


#save the data as csv

data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('dimerV_sapt.csv', data, delimiter=',', header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')





