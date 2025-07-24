

#import required modules

import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#load the data in

#the first ~14 frequencies from the QM data are imaginary numbers
#remove values from ir intensity, reduced masses, frequencies

coulomb_matrices = np.load('.npy')
#flatten coulomb matrix
coulomb_matrices = coulomb_matrices.reshape(1043, -1)  # Flatten
print("coulomb matrix shape",coulomb_matrices.shape)

dipoles = np.load('')
dipoles = dipoles.reshape(-1, 1)
print("dipoles shape", dipoles.shape)

ir_intensity = np.load('y')
ir_intensity = ir_intensity[:, 14:]
print("ir intensity", ir_intensity.shape)

energies = np.load('')
energies = energies.reshape(-1, 1)
print("energies shape", energies.shape)

reduced_masses = np.load('/y')
reduced_masses=reduced_masses[:,14:]
print("reduced masses shape", reduced_masses.shape)

frequencies=np.genfromtxt('',delimiter=',',dtype='str')
# Keep only columns from column 15 onward
frequencies = frequencies[:, 14:]
print("frequencies shape", frequencies.shape)

#use these in the graph along with the geometries
h7_distances=np.load('.npy')
h8_distances=np.load('/.npy')

#reshape
h7_distances=h7_distances.reshape(1043,1)
h8_distances=h8_distances.reshape(1043,1)

print("h7 distances", h7_distances.shape)
print("h8 distances", h8_distances.shape)



#clean the data
#define the preprocessing function
def preprocess(value):
    try:
        # Remove parentheses, convert to complex, and extract the real part
        complex_value = complex(value.strip("()"))  # Convert to complex
        return complex_value.real  # Extract real part
    except (ValueError, AttributeError):  # Handle invalid entries
        return np.nan  # Replace invalid values with NaN

frequencies_cleaned = np.vectorize(preprocess)(frequencies)

#handle NaN (e.g., replace with column means)
col_means = np.nanmean(frequencies_cleaned, axis=0)
indices = np.where(np.isnan(frequencies_cleaned))
frequencies_cleaned[indices] = np.take(col_means, indices[1])

print(frequencies_cleaned.shape)
#assign the cleaned data back to the original array name
frequencies = frequencies_cleaned

#scale the frequencies
scaling_factor = 0.975  # Typical for B3LYP/cc-pVDZ
frequencies = frequencies * scaling_factor

#save all arrays as np.float64 for precision
coulomb_matrices = coulomb_matrices.astype(np.float64)
dipoles = dipoles.astype(np.float64)
ir_intensity = ir_intensity.astype(np.float64)
reduced_masses = reduced_masses.astype(np.float64)
#h7_distances = h7_distances.astype(np.float64)
#h8_distances = h8_distances.astype(np.float64)
energies = energies.astype(np.float64)
frequencies = frequencies.astype(np.float64)

#combine the features taking out the angles as they are calculated further on
#do we want to include distances?
qm_features = np.concatenate((coulomb_matrices, dipoles, ir_intensity, reduced_masses,frequencies, energies), axis=1)
print("qm feature shape", qm_features.shape)

#load the geometries in

geometries = np.load('', allow_pickle=True)
all_geometries=geometries
print(all_geometries.shape)

#print the first geometry in all_geometries
print(all_geometries[0])



#calculate the angle between each hydrogen on the water and O1 carbonyl
#double check that the angles are correct

def calculate_angle(a, b, c):

    #define vectors BA and BC
    BA = a - b  # Vector from B (H7) to A (O1)
    BC = c - b  # Vector from B (H7) to C (O2)

    #compute dot product and magnitudes
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)

    #compute angle in radians and convert to degrees
    cosine_angle = dot_product / (magnitude_BA * magnitude_BC)
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure numerical stability
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

angles_o1_h7_o2 = []
angles_o1_h8_o2 = []

for geom in all_geometries:
    O1 = np.array([float(x) for x in geom[0, 1:]])
    H7 = np.array([float(x) for x in geom[11, 1:]])
    O2 = np.array([float(x) for x in geom[10, 1:]])
    H8 = np.array([float(x) for x in geom[12, 1:]])

    #angle O1-H7-O2
    angle_h7 = calculate_angle(O1, H7, O2)
    angles_o1_h7_o2.append(angle_h7)

    #angle O1-H8-O2
    angle_h8 = calculate_angle(O1, H8, O2)
    angles_o1_h8_o2.append(angle_h8)

#convert to numpy arrays
angles_o1_h7_o2 = np.array(angles_o1_h7_o2)
angles_o1_h8_o2 = np.array(angles_o1_h8_o2)

print(qm_features.shape)
print(angles_o1_h7_o2.shape)
print(angles_o1_h8_o2.shape)

#adding the angles to the qm_features
#do we want to include the angles in graph or in the features?

#qm_features = np.concatenate((qm_features, angles_o1_h7_o2.reshape(-1, 1), angles_o1_h8_o2.reshape(-1, 1)), axis=1)
#print(qm_features.shape)

#filter out the geometries with useful carbonyl peaks
carbonyl_freqs = []
geometry_indices = []

#identify geometries with good carbonyl match
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
#distances_h7_filtered = distances_h7[geometry_indices]
#distances_h8_filtered = distances_h8[geometry_indices]
o1_h7_o2_filtered = angles_o1_h7_o2[geometry_indices]
o1_h8_o2_filtered = angles_o1_h8_o2[geometry_indices]

#check the dimensions of the features
print("Filtered other frequencies shape:", frequencies_cleaned_filtered.shape)
print("Filtered IR shape:", ir_intensity_cleaned_filtered.shape)
print("filtered dipoles shape",dipoles_filtered.shape)
print("Filtered energies shape",energies_filtered.shape)
print("Filtered geometries shape",all_geometries_filtered.shape)
print("Filtered coulomb matrices shape",coulomb_matrices_filtered.shape)

#import the 10 experimental values
experimental_peaks = np.array([
    1728.9990040673013, 1696.9439484537018, 1697.0947727914245, 1697.3760126912327,
    1697.4989848742132, 1698.5709617481805, 1699.0717186046627, 1700.1131163109806,
    1701.003928111602, 1701.8026388562123
])

#create the delta matrix using the experimental peaks
delta_matrix = experimental_peaks - carbonyl_freqs[:, np.newaxis]

print(delta_matrix.shape)
print(delta_matrix[100])

#reshaping

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
    coulomb_matrices_filtered,
), axis=1)

print(qm_features.shape)

#import modules to create the graph 

import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

#compute angles. check same as above? 
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

#normalise the QM features 
print(qm_features.shape)  
scaler = StandardScaler()
qm_features_normalized = scaler.fit_transform(qm_features)

#save the scaler for use later
joblib.dump(scaler, "/content/drive/My Drive/scaler.save")

#atomic number lookup
atomic_numbers = {"H": 1, "C": 6, "O": 8}

#define the bonding for the graph, bidirectional
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
    (10, 11), (11, 10),  # O2 ↔ H7 (water)
    (10, 12), (12, 10),  # O2 ↔ H8 (water)
    (0, 11), (11, 0),    # O1 ↔ H7 (H-bond)
    (0, 12), (12, 0)     # O1 ↔ H8 (H-bond)
]

#build the graph

graph_dataset = []

for i, geom in enumerate(all_geometries_filtered):
    # atoms
    atom_numbers = [atomic_numbers[atom[0][0]] for atom in geom]
    coordinates = np.array([[float(x), float(y), float(z)] for _, x, y, z in geom])

    # node features
    node_features = torch.tensor(
        [[atom_numbers[j]] + coordinates[j].tolist() for j in range(len(atom_numbers))],
        dtype=torch.float
    )

    # edges 
    edge_index = torch.tensor(bond_definitions, dtype=torch.long).t().contiguous()
    bond_lengths = [np.linalg.norm(coordinates[src] - coordinates[dst]) for src, dst in bond_definitions]
    is_carbonyl = [1.0 if (src, dst) in [(0, 1), (1, 0)] else 0.0 for (src, dst) in bond_definitions]

    edge_attr = torch.tensor(bond_lengths, dtype=torch.float).view(-1, 1)
    carbonyl_flags = torch.tensor(is_carbonyl, dtype=torch.float).view(-1, 1)
    edge_attr = torch.cat((edge_attr, carbonyl_flags), dim=1)

    # hydrogen bond angles as features
    O1 = coordinates[0]    # acetone carbonyl oxygen
    O2 = coordinates[10]   # water oxygen
    H7 = coordinates[11]
    H8 = coordinates[12]
    angle_H7_O2_O1 = compute_angle(H7, O2, O1)
    angle_H8_O2_O1 = compute_angle(H8, O2, O1)

    # qm features 
    qm_feat = torch.tensor(qm_features_normalized[i], dtype=torch.float)

    # water coordinates
    water_coords = coordinates[[10, 11, 12]]  # shape: [3 atoms, 3 xyz] = [3, 3]
    water_coords_flat = torch.tensor(water_coords.reshape(-1), dtype=torch.float)  # shape: [9]

    
    delta_target = torch.tensor(delta_matrix[i], dtype=torch.float)

    # graph layout 
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        qm_features=qm_feat,
        y=water_coords_flat,  # 
        cond=qm_feat.view(1, -1), 
        hbond_angle=torch.tensor([angle_H7_O2_O1], dtype=torch.float),
        hbond_angle_alt=torch.tensor([angle_H8_O2_O1], dtype=torch.float),
        delta=delta_target  # Optional: store full delta vector if needed
    )

    # center on acetone oxygen
    origin_shift = graph_data.x[0, 1:]         # x, y, z of acetone oxygen
    graph_data.x[:, 1:] -= origin_shift        # shift all atoms
    graph_data.y = graph_data.y.view(3, 3) - origin_shift  # shift water target geometry too
    graph_data.y = graph_data.y.view(-1)       

    graph_dataset.append(graph_data)


#save the graph 
print(f"graphs created {len(graph_dataset)} graphs.")
torch.save(graph_dataset, )
print("graph dataset saved ")
