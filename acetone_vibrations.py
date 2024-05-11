#predict the vibrational frequencies of the acetone molecule using psi4

import psi4

acetone = psi4.geometry("""
0 1
   O        0.00024       -1.27917        0.00282
   C       -0.00000       -0.05752        0.00081
   C        1.29743        0.69092       -0.00041
   C       -1.29773        0.69041       -0.00042
   H        1.35910        1.32941       -0.90637
   H        1.35909        1.33238        0.90344
   H        2.15719       -0.01276        0.00075
   H       -2.15722       -0.01360        0.00073
   H       -1.35965        1.32887       -0.90638
   H       -1.35965        1.33185        0.90343

""")

#set memory
psi4.set_memory('4 GB')

psi4.set_options({'basis': 'cc-pvdz',
                    'scf_type': 'df',
                    'g_convergence': 'gau_tight',
                    'freeze_core': 'true'})

energy,wave_function = psi4.optimize('b3lyp/cc-pvdz',molecule=acetone,return_wfn=True)

frequency=psi4.frequencies('b3lyp/cc-pvdz',ref_gradient=wave_function.gradient(),molecule=acetone)

print(frequency)