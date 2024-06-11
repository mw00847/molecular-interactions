import autode as ade

water=ade.Molecule("water.xyz")
acetone=ade.Molecule("acetone.xyz")

#print the number of atoms
print("number of atoms", water.n_atoms)

H_water = water.atoms[1]
O_acetone = acetone.atoms[0]

# Translate the acetone molecule so that the oxygen atom is at the origin
acetone.translate(vec=-O_acetone.coord)

# Translate the water molecule so that the hydrogen atom is at the origin
water.translate(vec=-H_water.coord)

#print the new coordinates
print("after translation", water.coordinates)
print("after translation", acetone.coordinates)



