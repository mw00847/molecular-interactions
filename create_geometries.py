from define_model import ConditionalGraphVAE
import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 

def fix_acetone_atoms(recon_x, acetone_template):

    batch_size = recon_x.shape[0]
    acetone_repeated = acetone_template.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 10, 4]

    # Combine acetone with predicted water (10:13)
    water = recon_x[:, 10:13, :].clone()  # detach water part safely
    fixed = torch.cat([acetone_repeated, water], dim=1)
    return fixed


#define bond angles and lengths

def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)*10

#compute bond angle
def bond_angle(atom1, atom2, atom3):
    vec1 = atom1 - atom2
    vec2 = atom3 - atom2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure it's in range
    return np.degrees(angle)*1000  # Convert to degrees

# fix acetone geometry for atoms 0–9 (shape: [10, 4])
acetone_template = torch.tensor([
    [8, 0.00000, 0.00000, 0.00000],
    [6, -0.18923, 0.84171, 0.86495],
    [6, 0.96812, 1.38841, 1.64288],
    [6, -1.57836, 1.32603, 1.14647],
    [1, 1.03349, 2.48631, 1.49265],
    [1, 0.82576, 1.17534, 2.72296],
    [1, 1.92064, 0.92411, 1.30898],
    [1, -2.31283, 0.82041, 0.48370],
    [1, -1.63426, 2.42096, 0.97259],
    [1, -1.84199, 1.11000, 2.20291],
], device=device)



# load dataset
dataset = torch.load("graph_dataset_with_H8.pt")

# split into train, val, test
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

#define the model
model = ConditionalGraphVAE(
    in_channels=4,
    cond_dim=10,
    hidden_dim=64,
    latent_dim=32,
    out_channels=4
).to(device)

#load checkpoint
checkpoint = torch.load("50000_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


sample = test_data[0].to(device)
sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)

bond_lengths_OH1 = []
bond_lengths_OH2 = []
bond_angles = []

for i in range(10):
    condition = torch.zeros((1, 10), device=device)
    condition[0, i] = i / 9  # Normalized concentration

    with torch.no_grad():
        mu, logvar = model.encoder(sample.x, sample.edge_index, sample.batch, condition)
        z = mu

        recon_x = model.decoder(z, sample.edge_index, sample.batch, condition)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)
        geometry = recon_x.squeeze(0).cpu().numpy()

        O = geometry[10, 1:]
        H1 = geometry[11, 1:]
        H2 = geometry[12, 1:]

        b1 = bond_length(O, H1)
        b2 = bond_length(O, H2)
        angle = bond_angle(H1, O, H2)

        bond_lengths_OH1.append(b1)
        bond_lengths_OH2.append(b2)
        bond_angles.append(angle)

        print(f"concentration {i/9:.2f} → O–H1: {b1:.4f} Å, O–H2: {b2:.4f} Å, H–O–H angle: {angle:.2f}°")
        print("=====================================")


# plot bond lengths
plt.figure(figsize=(10, 4))
plt.plot(bond_lengths_OH1, label='O–H1', marker='o')
plt.plot(bond_lengths_OH2, label='O–H2',marker='o')
plt.axhline(0.96, color='gray', linestyle='--', label='Target')
plt.title("O–H Bond Lengths vs Concentration")
plt.xlabel("Concentration Index")
plt.ylabel("Bond Length (Å)")
plt.legend()
plt.ylim(0, 3)
plt.grid(True)
plt.tight_layout()
plt.show()

#plot bond angle
plt.figure(figsize=(6, 4))
plt.plot(bond_angles, marker='o', label='H–O–H Angle')
plt.axhline(104.5, color='gray', linestyle='--', label='Target')
plt.title("Water Bond Angle vs Concentration")
plt.xlabel("Concentration Index")
plt.ylabel("Angle (°)")
plt.legend()
plt.ylim(0, 300)
plt.grid(True)
plt.tight_layout()
plt.show()

