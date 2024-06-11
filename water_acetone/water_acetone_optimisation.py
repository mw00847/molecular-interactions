import psi4
#print the psi4 version
print(psi4.__version__)

# Clean any previous Psi4 calculations
psi4.core.clean()

# Set Psi4 options
psi4.set_memory('4 GB')
psi4.set_num_threads(4)

# Define the molecular complex geometry
complex = psi4.geometry("""
0 1
   O       -0.13754       -0.66135        1.57282
   C        0.04817        0.14202        0.67143
   C        1.37504        0.82389        0.53855
   C       -1.05047        0.44243       -0.30115
   H        1.82015        0.58464       -0.44982
   H        1.24015        1.92252        0.62208
   H        2.07166        0.48677        1.33563
   H       -1.96064       -0.14736       -0.06032
   H       -0.71949        0.18521       -1.32902
   H       -1.30209        1.52270       -0.25805
--
   O        0.52170       -1.71790       -0.40484
   H        1.03616       -1.84868        0.45027
   H        0.26309       -2.45726       -1.15789
""")


psi4.set_module_options('optking', {'frozen_distance': '1 13'})


# Optimize the geometry
psi4.optimize('B3LYP/6-31G', molecule=complex)

# Print the optimized geometry
print(psi4.dump_molecule_out(complex))

