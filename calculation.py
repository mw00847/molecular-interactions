


#this looks at different isomers of phtalic acid and plots the elst, ind, disp and exchange decomposition energies using SAPT


import psi4 
import numpy as np

psi4.set_options({'scf_type': 'df',
                  'freeze_core': 'true'})
                  
#different combinations of the isomers 

#O/M = comparison1 
#O/P = comparison2 
#M/P = comparison3 

#comparison1 O/M               


comparison1 = psi4.geometry("""

0 1 

O          1.72900        1.44890        1.30290
O          0.52150       -2.76860        0.50990
O          1.94280        1.60280       -0.95480
O          1.94610       -1.35340       -0.55510
C         -0.07460        0.73300       -0.00740
C         -0.28440       -0.64490       -0.05810
C         -1.16380        1.60440        0.00750
C         -1.58340       -1.15170       -0.09400
C         -2.46290        1.09750       -0.02840
C         -2.67270       -0.28050       -0.07920
C          1.27000        1.29310        0.03190
C          0.83250       -1.58060       -0.07500
H         -1.01340        2.68020        0.04680
H         -1.78090       -2.21920       -0.14740
H         -3.31100        1.77580       -0.01770
H         -3.68430       -0.67460       -0.11040
H          2.63430        1.82640        1.32040
H          1.27850       -3.39240        0.49470

--

0 1 
O         -3.54720        0.18420        0.00030
O          3.54710        0.18450       -0.00070
O         -2.58510       -1.87520       -0.00050
O          2.58550       -1.87510        0.00060
C         -1.20810        0.08130        0.00030
C          1.20790        0.08110       -0.00010
C         -0.00020       -0.61630        0.00010
C         -1.20790        1.47630        0.00020
C          1.20790        1.47610       -0.00020
C          0.00000        2.17360        0.00010
C         -2.46790       -0.64530       -0.00040
C          2.46800       -0.64530        0.00040
H         -0.00020       -1.70390        0.00010
H         -2.12470        2.05900        0.00020
H          2.12470        2.05880       -0.00020
H          0.00010        3.25980        0.00000
H         -4.39140       -0.31530        0.00040
H          4.39150       -0.31500       -0.00100
""")

psi4.energy('sapt0/jun-cc-pvdz', molecule=comparison1)


one_disp = psi4.variable('SSAPT0 DISP ENERGY')
one_elst = psi4.variable('SSAPT0 ELST ENERGY')
one_exch = psi4.variable('SSAPT0 EXCH ENERGY')
one_ind = psi4.variable('SSAPT0 IND ENERGY')
one_tot =psi4.variable('SSAPT0 TOTAL ENERGY')



#comparison2 O/P
 
comparison2 = psi4.geometry("""
 
0 1
 
O          1.72900        1.44890        1.30290
O          0.52150       -2.76860        0.50990
O          1.94280        1.60280       -0.95480
O          1.94610       -1.35340       -0.55510
C         -0.07460        0.73300       -0.00740
C         -0.28440       -0.64490       -0.05810
C         -1.16380        1.60440        0.00750
C         -1.58340       -1.15170       -0.09400
C         -2.46290        1.09750       -0.02840
C         -2.67270       -0.28050       -0.07920
C          1.27000        1.29310        0.03190
C          0.83250       -1.58060       -0.07500
H         -1.01340        2.68020        0.04680
H         -1.78090       -2.21920       -0.14740
H         -3.31100        1.77580       -0.01770
H         -3.68430       -0.67460       -0.11040
H          2.63430        1.82640        1.32040
H          1.27850       -3.39240        0.49470
 
--
 
0 1 
O          3.39660        1.18370       -0.00030
O         -3.39650       -1.18360       -0.00070
O          3.54230       -1.08460        0.00020
O         -3.54230        1.08470        0.00000
C          1.39460       -0.03050       -0.00050
C         -1.39450        0.03050        0.00050
C          0.72350        1.19240       -0.00030
C         -0.67100        1.22300        0.00030
C          0.67090       -1.22310       -0.00020
C         -0.72370       -1.19250        0.00030
C          2.84840       -0.06240        0.00030
C         -2.84820        0.06250        0.00030
H          1.24760        2.14410       -0.00030
H         -1.17530        2.18580        0.00040
H          1.17520       -2.18590       -0.00010
H         -1.24770       -2.14420        0.00050
H          4.37700        1.15110       -0.00040
H         -4.37700       -1.15080       -0.00130
""")

psi4.energy('sapt0/jun-cc-pvdz', molecule=comparison2)



two_disp = psi4.variable('SSAPT0 DISP ENERGY')
two_elst = psi4.variable('SSAPT0 ELST ENERGY')
two_exch = psi4.variable('SSAPT0 EXCH ENERGY')
two_ind = psi4.variable('SSAPT0 IND ENERGY')
two_tot =psi4.variable('SSAPT0 TOTAL ENERGY')


#comparison3 M/P

comparison3 = psi4.geometry("""

0 1 
O         -3.54720        0.18420        0.00030
O          3.54710        0.18450       -0.00070
O         -2.58510       -1.87520       -0.00050
O          2.58550       -1.87510        0.00060
C         -1.20810        0.08130        0.00030
C          1.20790        0.08110       -0.00010
C         -0.00020       -0.61630        0.00010
C         -1.20790        1.47630        0.00020
C          1.20790        1.47610       -0.00020
C          0.00000        2.17360        0.00010
C         -2.46790       -0.64530       -0.00040
C          2.46800       -0.64530        0.00040
H         -0.00020       -1.70390        0.00010
H         -2.12470        2.05900        0.00020
H          2.12470        2.05880       -0.00020
H          0.00010        3.25980        0.00000
H         -4.39140       -0.31530        0.00040
H          4.39150       -0.31500       -0.00100

--
 
0 1 
O          3.39660        1.18370       -0.00030
O         -3.39650       -1.18360       -0.00070
O          3.54230       -1.08460        0.00020
O         -3.54230        1.08470        0.00000
C          1.39460       -0.03050       -0.00050
C         -1.39450        0.03050        0.00050
C          0.72350        1.19240       -0.00030
C         -0.67100        1.22300        0.00030
C          0.67090       -1.22310       -0.00020
C         -0.72370       -1.19250        0.00030
C          2.84840       -0.06240        0.00030
C         -2.84820        0.06250        0.00030
H          1.24760        2.14410       -0.00030
H         -1.17530        2.18580        0.00040
H          1.17520       -2.18590       -0.00010
H         -1.24770       -2.14420        0.00050
H          4.37700        1.15110       -0.00040
H         -4.37700       -1.15080       -0.00130
""")


psi4.energy('sapt0/jun-cc-pvdz', molecule=comparison3)


three_disp = psi4.variable('SSAPT0 DISP ENERGY')
three_elst = psi4.variable('SSAPT0 ELST ENERGY')
three_exch = psi4.variable('SSAPT0 EXCH ENERGY')
three_ind = psi4.variable('SSAPT0 IND ENERGY')
three_tot =psi4.variable('SSAPT0 TOTAL ENERGY')

#comparing the interaction energies for each dimer 

labels = np.array(['disp','elst','exch','ind','tot'])

dispersion = np.array([one_disp,two_disp,three_disp])

elst = np.array([one_elst,two_elst,three_elst])

exch = np.array([one_exch,two_exch,three_exch])

ind = np.array([one_ind,two_ind,three_ind])

ind = np.array([one_ind,two_ind,three_ind])

tot = np.array([one_tot,two_tot,three_tot])

#plotting these comparisons 

plt.plot(comparison,tot,'-ob')

plt.plot(comparison,ind,'-xr')

plt.plot(comparison,exch,'-py')

plt.plot(comparison,elst,'-dk')

plt.plot(comparison,dispersion,'-m<')

plt.show()


