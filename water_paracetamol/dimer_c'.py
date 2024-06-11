#original water z matrix
#    O   14 B20 6 A20 5 D20
#    H   21 B21 14 A21 6 D21
#    H   21 B22 14 A22 6 D22



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
    dimerc = psi4.geometry("""
    
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
    H   17 B23 14 A22 6 D22
    O   21 B22 6 A20 5 D20
    H   22 B21 14 A21 6 D21
    

    B1   =     1.38920
    B2   =     1.39875
    A2   =   119.65538
    B3   =     1.40433
    A3   =   121.22674
    D3   =     0.10816
    B4   =     1.39682
    A4   =   119.31596
    D4   =   359.92301
    B5   =     1.39520
    A5   =   121.24755
    D5   =   359.97121
    B6   =     1.41966
    A6   =   116.76891
    D6   =   179.25370
    B7   =     1.36714
    A7   =   130.01296
    D7   =   172.63432
    B8   =     1.52201
    A8   =   114.83878
    D8   =   180.40855
    B9   =     1.09456
    A9   =   114.32900
    D9   =     3.57696
    B10  =      1.09488
    A10  =    108.69982
    D10  =    241.86241
    B11  =      1.09404
    A11  =    108.60294
    D11  =    125.46205
    B12  =      1.08859
    A12  =    120.04362
    D12  =    180.34564
    B13  =      1.08101
    A13  =    118.89748
    D13  =    180.32679
    B14  =      1.08505
    A14  =    121.03813
    D14  =    179.97388
    B15  =      1.36826
    A15  =    117.60948
    D15  =    179.97059
    B16  =      1.22870
    A16  =    124.52683
    D16  =      0.51178
    B17  =      1.08868
    A17  =    119.05914
    D17  =    180.12945
    B18  =      0.96989
    A18  =    108.81875
    D18  =    179.34723
    B19  =      1.01053
    A19  =    114.22600
    D19  =    354.54930
    B20  =      2.37986
    A20  =    170.09776
    D20  =      1.00145
    B21  =      0.96910
    A21  =     96.62498
    D21  =    275.62051
    B22  =      0.97472
    A22  =     61.57173
    D22  =    174.82511
    
    
    B23 = """ + str(distances[i]) + """

  """)

    psi4.set_options({'scf_type': 'df',
                      'freeze_core': 'true',
                      })
    #
    # # save the plotsapt calculation

    psi4.energy('sapt0/jun-cc-pvdz', molecule=dimerc)

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
plt.savefig('dimerc_sapt.png')

# save the data as csv

data = np.column_stack((distances, eelst, eexch, eind, edisp, esapt))
np.savetxt('dimerc_sapt.csv', data, delimiter=',',
           header='Distance,Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')
#plt.show()



