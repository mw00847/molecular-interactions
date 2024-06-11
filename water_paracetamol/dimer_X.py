import psi4
import numpy as np
import matplotlib.pyplot as plt

psi4.core.clean()

# Set memory & output
psi4.set_memory('10 GB')

# distances is a list between 1 and 3 with steps of 0.1

distances = np.arange(1, 2.5, 0.1)

# Define arrays for each energy component
eelst = np.zeros((len(distances)))
eexch = np.zeros((len(distances)))
eind = np.zeros((len(distances)))
edisp = np.zeros((len(distances)))
esapt = np.zeros((len(distances)))

for i in range(len(distances)):
    dimerX = psi4.geometry("""
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
    
    B1 =       1.38986
    B2 =       1.40109
    A2 =     120.13049
    B3 =       1.40367
    A3 =     121.08000
    D3 =     359.99352
    B4 =       1.40112
    A4 =     118.89212
    D4 =       0.00625
    B5 =       1.39483
    A5 =     121.08387
    D5 =     359.99859
    B6 =       1.41740
    A6 =     117.53921
    D6 =     179.99468
    B7 =       1.37363
    A7 =     128.99617
    D7 =     179.93426
    B8 =       1.52506
    A8 =     114.44485
    D8 =     180.17618
    B9 =       1.09493
    A9 =     114.37479
    D9 =     358.68724
    B10 =       1.09445
    A10 =     108.65577
    D10 =     236.89112
    B11 =       1.09466
    A11 =     108.71384
    D11 =     120.46608
    B12 =       1.08710
    A12 =     119.48581
    D12 =     180.00034
    B13 =       1.08095
    A13 =     120.75511
    D13 =     180.00231
    B14 =       1.08565
    A14 =     120.87789
    D14 =     180.00427
    B15 =       1.36080
    A15 =     117.96614
    D15 =     180.00690
    B16 =       1.22499
    A16 =     124.36831
    D16 =       0.07438
    B17 =       1.08903
    A17 =     119.20066
    D17 =     180.00022
    B18 =       0.98227
    A18 =     109.68249
    D18 =     180.00832
    B19 =       1.01018
    A19 =     114.67990
    D19 =       0.02167
    
    --
    
    O   19 B20 16 A20 3 D20
    H   21 B21 19 A21 16 D21
    H   21 B22 19 A22 16 D22
    
    
    
    B20 =       """ + str(distances[i]) + """
    A20 =     174.25161
    D20 =     181.98237
    B21 =       0.97002
    A21 =     108.86759
    D21 =      55.36655
    B22 =       0.96993
    A22 =     109.07657
    D22 =     301.21388

    
    
    
    """)

    psi4.set_options({'scf_type': 'df',
                      'freeze_core': 'true',
                      })
    # sapt calculation

    psi4.energy('sapt0/jun-cc-pvdz', molecule=dimerX)

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

# save the plot

plt.savefig('dimerX_sapt.png')

# save the data as csv

data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('dimerX_sapt.csv', data, delimiter=',',
           header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')