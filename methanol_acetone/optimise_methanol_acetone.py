#optimise the geometry of methanol and acetone

import psi4

#geometry

dimer = psi4.geometry("""
O 4.02912 1.04732 -0.00014
C 4.02882 2.27722 0.00066
C 5.30982 3.06682 -0.00014
C 2.74752 3.06632 -0.00014
H 5.35672 3.68792 -0.89794
H 5.35482 3.69262 0.89456
H 6.16392 2.38402 0.00276
H 1.89362 2.38312 0.00276
H 2.70042 3.68742 -0.89794
H 2.70222 3.69222 0.89456
C 0.52071 -0.60873 -0.00020
O 1.59071 -0.60873 -0.00020
H 0.16405 -0.01556 0.81579
H 0.16404 -1.61199 0.10551
H 0.16404 -0.19866 -0.92190
H 1.91404 -0.98048 0.83536                      
""")

psi4.set_options({'scf_type': 'df',
                    'freeze_core': 'true'})

#optimse molecule using jun-cc-pvdz basis set

psi4.optimize('b3lyp/6-31g', molecule=dimer)




#print the coordinates of the optimised geometry

xyz_geometry = dimer.to_string(dtype='xyz')
print(f"XYZ Geometry for optimised geometry:\n{xyz_geometry}\n")

