#use autode to make H3 the origin
#https://duartegroup.github.io/autodE/examples/molecules.html

import autode as ade

# Define the molecule
water = ade.Molecule(name='h2o', smiles='O')
print("before translation", water.coordinates)

h_atom = water.atoms[2]
water.translate(vec=-h_atom.coord)
print("after translation", water.coordinates)