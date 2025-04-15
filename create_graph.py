#CREATE A GRAPH FOR ML


#import modules
import numpy as np
import pandas as pd

#load the geometries in

import importlib.util
import numpy as np

def load_geometries(module_name, module_path):
    """Dynamically load a Python module from a given path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#paths to the geometry files
geometry_path1 = ""
geometry_path2 = ""

#load the geomtries
geometries1 = load_geometries("geometries_named1", geometry_path1)
geometries2 = load_geometries("geometries_named2", geometry_path2)

#create lists to store all geometries
all_geometries = []

#extract geometries from the first file (geometries1)
for i in range(1, 500):  # 499 geometries
    geometry_var = f"geometry_{i}"
    if hasattr(geometries1, geometry_var):  # Check if variable exists
        all_geometries.append(getattr(geometries1, geometry_var))

#extract geometries from the second file (geometries2)
for i in range(1, 500):  # 499 geometries
    geometry_var = f"geometry_{i}"
    if hasattr(geometries2, geometry_var):  # Check if variable exists
        all_geometries.append(getattr(geometries2, geometry_var))

#save the geometries as all_geometries
all_geometries = np.array(all_geometries, dtype=object)

print(f"concatenated {len(all_geometries)} geometries.")

#import the rest of the data
dipoles=np.load('dipoles.npy')
dipoles=dipoles.reshape(-1)
energies=np.load('energies.npy')
energies=energies.reshape(-1)
ir_intensity=np.load('ir_intensity.npy')
frequencies=np.load('frequencies.npy')
coulomb_matrices=np.load('coulomb_matrices.npy')

#print the shapes of arrays
print(dipoles.shape)
print(energies.shape)
print(ir_intensity.shape)
print(frequencies.shape)
print(coulomb_matrices.shape)

#import the calculated distances between o1 and H7,H8

distances_h7=np.load('O1-H7_Distances.npy')
distances_h8=np.load('O1-H8_Distances.npy')


#filter out the geometries with useful carbonyl peaks
carbonyl_freqs = []
geometry_indices = []


for geom_idx in range(frequencies.shape[0]):
    freqs = frequencies[geom_idx]
    irs = ir_intensity[geom_idx]

    mask = (freqs >= 1600) & (freqs <= 1850) & (irs > 50)
    candidate_freqs = freqs[mask]
    candidate_irs = irs[mask]

    if candidate_freqs.size > 0:
        deltas = np.abs(candidate_freqs - 1709)
        best_idx = np.argmin(deltas)
        best_freq = candidate_freqs[best_idx]
        carbonyl_freqs.append(best_freq)
        geometry_indices.append(geom_idx)

#convert to arrays
carbonyl_freqs = np.array(carbonyl_freqs)
geometry_indices = np.array(geometry_indices)

#use those indices to filter the features
frequencies_cleaned_filtered = frequencies[geometry_indices]
ir_intensity_cleaned_filtered = ir_intensity[geometry_indices]
dipoles_filtered = dipoles[geometry_indices]
energies_filtered = energies[geometry_indices]
all_geometries_filtered = all_geometries[geometry_indices]
coulomb_matrices_filtered = coulomb_matrices[geometry_indices]
distances_h7_filtered = distances_h7[geometry_indices]
distances_h8_filtered = distances_h8[geometry_indices]

#check the dimensions of the features
print("Filtered other frequencies shape:", frequencies_cleaned_filtered.shape)
print("Filtered IR shape:", ir_intensity_cleaned_filtered.shape)
print("filtered dipoles shape",dipoles_filtered.shape)
print("Filtered energies shape",energies_filtered.shape)
print("Filtered geometries shape",all_geometries_filtered.shape)
print("Filtered coulomb matrices shape",coulomb_matrices_filtered.shape)
print("Filtered H7 distances shape",distances_h7_filtered.shape)
print("Filtered H8 distances shape",distances_h8_filtered.shape)

#import the 10 experimental values
experimental_peaks = np.array([
    1728.9990040673013, 1696.9439484537018, 1697.0947727914245, 1697.3760126912327,
    1697.4989848742132, 1698.5709617481805, 1699.0717186046627, 1700.1131163109806,
    1701.003928111602, 1701.8026388562123
])

#delta matrix is the difference between the experimental carbonyl peak and the QM frequencies for each geometry
delta_matrix = experimental_peaks - carbonyl_freqs[:, np.newaxis]

print(delta_matrix.shape)
print(delta_matrix[100])

dipoles_filtered = dipoles_filtered.reshape(-1, 1) if dipoles_filtered.ndim == 1 else dipoles_filtered
energies_filtered = energies_filtered.reshape(-1, 1) if energies_filtered.ndim == 1 else energies_filtered
frequencies_cleaned_filtered = frequencies_cleaned_filtered.reshape(-1, 1) if frequencies_cleaned_filtered.ndim == 1 else frequencies_cleaned_filtered
ir_intensity_cleaned_filtered = ir_intensity_cleaned_filtered.reshape(-1, 1) if ir_intensity_cleaned_filtered.ndim == 1 else ir_intensity_cleaned_filtered
coulomb_matrices_filtered = coulomb_matrices_filtered.reshape(-1, 1) if coulomb_matrices_filtered.ndim == 1 else coulomb_matrices_filtered

qm_features = np.concatenate((
    dipoles_filtered,
    energies_filtered,
    frequencies_cleaned_filtered,
    ir_intensity_cleaned_filtered,
    coulomb_matrices_filtered
), axis=1)

#install torch_geometric used to create the graph
!pip install torch_geometric torch numpy

#create the graphs
import torch
from torch_geometric.data import Data
import numpy as np
import importlib.util
from sklearn.preprocessing import StandardScaler

#concatenate the QM features
qm_features = np.concatenate((dipoles_filtered, energies_filtered,frequencies_cleaned_filtered,ir_intensity_cleaned_filtered,coulomb_matrices_filtered), axis=1)

#normalize the QM Features
scaler = StandardScaler()
qm_features_normalized = scaler.fit_transform(qm_features)

#map the atomic numbers
atomic_numbers = {"H": 1, "C": 6, "O": 8}

#declare the bonding (excluding H7 ↔ O1 initially)
bond_definitions = [
    (0, 1), (1, 0),  # O1 ↔ C1
    (1, 2), (2, 1),  # C1 ↔ C2
    (1, 3), (3, 1),  # C1 ↔ C3
    (2, 4), (4, 2),  # C2 ↔ H1
    (2, 5), (5, 2),  # C2 ↔ H2
    (2, 6), (6, 2),  # C2 ↔ H3
    (3, 7), (7, 3),  # C3 ↔ H4
    (3, 8), (8, 3),  # C3 ↔ H5
    (3, 9), (9, 3),  # C3 ↔ H6
    (10, 11), (11, 10),  # O2 ↔ H7
    (10, 12), (12, 10),  # O2 ↔ H8
]

#declare the hydrogen bonds O1 ↔ H7 & O1 ↔ H8
bond_definitions += [(0, 11), (11, 0)]  # O1 ↔ H7
bond_definitions += [(0, 12), (12, 0)]  # O1 ↔ H8

#create graph
graph_dataset = []

for i, geom in enumerate(all_geometries_filtered):
    #extract atomic numbers and coordinates
    atom_numbers = [atomic_numbers[atom[0][0]] for atom in geom]
    coordinates = np.array([[float(x), float(y), float(z)] for _, x, y, z in geom])

    #create node features (atomic number + XYZ)
    node_features = torch.tensor(
        [[atom_numbers[i]] + coordinates[i].tolist() for i in range(len(atom_numbers))],
        dtype=torch.float
    )

    #convert edge list to pytorch
    edge_index = torch.tensor(bond_definitions, dtype=torch.long).t().contiguous()

    #create binary feature for the carbonyl bond (O1 ↔ C1)
    is_carbonyl = [1.0 if bond in [(0, 1), (1, 0)] else 0.0 for bond in bond_definitions]
    carbonyl_flags = torch.tensor(is_carbonyl, dtype=torch.float).view(-1, 1)

    #add the hydrogen bond distances
    o1_h7_distance = distances_h7_filtered[i]
    o1_h8_distance = distances_h8_filtered[i]

    #modify edges with the hydrogen bond distances
    edge_attr_values = [1.0] * (len(edge_index.T) - 4)
    edge_attr_values += [o1_h7_distance, o1_h7_distance]
    edge_attr_values += [o1_h8_distance, o1_h8_distance]

    #convert to pytorch tensors with carbonyl binaries
    edge_attr = torch.tensor(edge_attr_values, dtype=torch.float).view(-1, 1)
    edge_attr = torch.cat((edge_attr, carbonyl_flags), dim=1)

    #extract QM Features and separate the target
    qm_feat = torch.tensor(qm_features_normalized[i], dtype=torch.float)

    #loop through the delta matrix as the target
    y_target = torch.tensor(delta_matrix[i], dtype=torch.float)  # Shape: [10]

    #create graph data object
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        qm_features=qm_feat,
        y=y_target
    )

    #append graph to dataset
    graph_dataset.append(graph_data)

#check dataset length before saving
print(f"created {len(graph_dataset)} graphs.")

#save the updated dataset
torch.save(graph_dataset, "graph_dataset_with_H8.pt")

print("graph dataset saved successfully!")

#making sure the data is correct

import torch
from scipy.spatial.distance import euclidean

from scipy.spatial.distance import euclidean

for i, graph in enumerate(graph_dataset):
    coords = graph.x[:, 1:].numpy()  # extract XYZ
    atom1 = coords[11]  # index 0 = atom 1
    atom2 = coords[12]  # index 1 = atom 2
    distance = euclidean(atom1, atom2)
    print(f"graph {i}: distance between atom 1 and 2 = {distance:.4f} Å")