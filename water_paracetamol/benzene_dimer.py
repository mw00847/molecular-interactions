#benzene dimer with optimized geometry UFF

import psi4
import numpy as np

psi4.core.clean()

# Set memory & output
psi4.set_memory('10 GB')


benzene_dimer = psi4.geometry("""
    0 1
    H  
    C   1 B1
    C   2 B2 1 A2
    H   3 B3 2 A3 1 D3
    C   3 B4 2 A4 1 D4
    H   5 B5 3 A5 2 D5
    C   5 B6 3 A6 2 D6
    H   7 B7 5 A7 3 D7
    C   7 B8 5 A8 3 D8
    H   9 B9 7 A9 5 D9
    C   9 B10 7 A10 5 D10
    H   11 B11 9 A11 7 D11
    --
    H   1 B12 2 A12 3 D12
    C   13 B13 1 A13 2 D13
    C   14 B14 13 A14 1 D14
    H   15 B15 14 A15 13 D15
    C   15 B16 14 A16 13 D16
    H   17 B17 15 A17 14 D17
    C   17 B18 15 A18 14 D18
    H   19 B19 17 A19 15 D19
    C   19 B20 17 A20 15 D20
    H   21 B21 19 A21 17 D21
    C   14 B22 13 A22 1 D22
    H   23 B23 14 A23 13 D23
    
    B1   =     1.08222
    B2   =     1.39887
    A2   =   119.99984
    B3   =     1.08222
    A3   =   120.00019
    D3   =     0.00104
    B4   =     1.39887
    A4   =   120.00001
    D4   =   179.90492
    B5   =     1.08222
    A5   =   119.99995
    D5   =   180.09686
    B6   =     1.39886
    A6   =   120.00012
    D6   =     0.00003
    B7   =     1.08222
    A7   =   119.99975
    D7   =   180.09661
    B8   =     1.39887
    A8   =   120.00011
    D8   =   359.99997
    B9   =     1.08222
    A9   =   119.99975
    D9   =   180.09554
    B10  =      1.39887
    A10  =    119.99992
    D10  =      0.00004
    B11  =      1.08222
    A11  =    119.99994
    D11  =    180.09485
    B12  =      3.51397
    A12  =     89.69686
    D12  =    270.41412
    B13  =      1.08222
    A13  =     90.46912
    D13  =      0.00002
    B14  =      1.39887
    A14  =    120.00006
    D14  =     90.31831
    B15  =      1.08222
    A15  =    119.99979
    D15  =      0.00104
    B16  =      1.39887
    A16  =    119.99997
    D16  =    180.09658
    B17  =      1.08222
    A17  =    119.99990
    D17  =    179.90515
    B18  =      1.39887
    A18  =    119.99988
    D18  =      0.00001
    B19  =      1.08222
    A19  =    120.00018
    D19  =    179.90490
    B20  =      1.39887
    A20  =    119.99991
    D20  =    359.99999
    B21  =      1.08222
    A21  =    120.00019
    D21  =    179.90394
    B22  =      1.39886
    A22  =    119.99981
    D22  =    270.41488
    B23  =      1.08222
    A23  =    119.99983
    D23  =      0.00029


""")

psi4.set_options({'scf_type': 'df',
                  'freeze_core': 'true',
                  })
# sapt calculation in kcal/mol


psi4.energy('sapt0/jun-cc-pvdz', molecule=benzene_dimer)
elst = psi4.variable('SAPT ELST ENERGY')*627.509
exch = psi4.variable('SAPT EXCH ENERGY')*627.509
ind = psi4.variable('SAPT IND ENERGY')*627.509
disp = psi4.variable('SAPT DISP ENERGY')*627.509
tot = psi4.variable('SAPT TOTAL ENERGY')*627.509

psi4.core.clean()

#plotting the energies
import matplotlib.pyplot as plt
import numpy as np

#data to plot
n_groups = 5

labels = np.array(['disp','elst','exch','ind','tot'])

values = np.array([disp,elst,exch,ind,tot])

#plotting
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
#disp red, elst green, exch blue, ind yellow, tot purple
plt.bar(index, values, color=['r','g','b','y','m'])
plt.xlabel('Energy Type')
plt.ylabel('Energy (kcal/mol)')
plt.title('SAPT0 Energy Decomposition benzene from Avogadro')
plt.xticks(index , labels)
plt.tight_layout()
plt.show()

#save the plot

plt.savefig('benzene_dimer_sapt.png')


#save the data as csv

data = np.column_stack((disp,elst,exch,ind,tot))
np.savetxt('benzene_dimer_sapt.csv', data, delimiter=',', header='Electrostatics,Exchange,Induction,Dispersion,SAPT Total', comments='')


