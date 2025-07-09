

"""
this model uses a conditional graph autoencoder (VAE) with PyGeometric to generate water molecules around a fixed acetone.

the VAE is conditioned on 221 dimensional dataset made up of QM/IR with 1043 samples

"""


#import modules 

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim


#define the device as cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the graph dataset 
dataset=torch.load("   ",weights_only=False)

#use a class to use one vector for each graph 
from torch_geometric.data import Data

class GraphWithCond(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cond":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

#split the data into train and validation

#print the graph keys
print(f"graph keys: {dataset[0].keys()}")  #'x', 'edge_index', 'qm_features', 'y'

#define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

#train/val/test split
train_data, temp_data = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

from torch_geometric.loader import DataLoader

#dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#encoder uses transformer graph attention mechanism, outputs both mu and logvar 

from torch_geometric.nn import TransformerConv

class ConditionalTransformerEncoder(nn.Module):
    def __init__(self, in_channels, cond_dim, hidden_dim, latent_dim, heads=1):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.conv2 = TransformerConv(hidden_dim * heads, latent_dim, heads=1, concat=False)
        self.fc_cond = nn.Linear(cond_dim, latent_dim)
        self.mu_layer = TransformerConv(latent_dim, latent_dim)
        self.logvar_layer = TransformerConv(latent_dim, latent_dim)

    def forward(self, x, edge_index, batch, condition):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        cond_embed = self.fc_cond(condition)
        cond_expanded = cond_embed[batch]
        x = x + cond_expanded
        x = torch.clamp(x, -10, 10)
        mu = self.mu_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        logvar = torch.clamp(logvar, min=-5, max=5)
        return mu, logvar

#decoder, reconstructs only the water atoms 

class ConditionalTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, out_channels=3, heads=1):
        super().__init__()
        self.fc_cond = nn.Linear(cond_dim, latent_dim)
        self.conv1 = TransformerConv(latent_dim * 2, hidden_dim, heads=heads, concat=True)
        self.conv2 = TransformerConv(hidden_dim * heads, out_channels, heads=1, concat=False)

    def forward(self, z, edge_index, batch, condition):
        #conditions [batch_size, cond_dim]

        cond_embed = self.fc_cond(condition)        # batch_size, latent_dim
        cond_expanded = cond_embed[batch]           # num_nodes, latent_dim
        z = torch.cat([z, cond_expanded], dim=1)    # num_nodes, latent_dim * 2

        x = F.elu(self.conv1(z, edge_index))        # num_nodes, hidden_dim * heads
        x = self.conv2(x, edge_index)               #num_nodes, 3

        #extract predicted water atom positions
        
	num_nodes_per_graph = 13
        batch_size = batch.max().item() + 1
        
	#specify the water atoms 
	water_indices = (
            torch.arange(batch_size, device=z.device).repeat_interleave(3) * num_nodes_per_graph
            + torch.tensor([10, 11, 12], device=z.device).repeat(batch_size)
        )
        x = x[water_indices].view(batch_size, 3, 3)
        return x

#describe the VAE 

class ConditionalGraphVAE(nn.Module):
    def __init__(self, in_channels, cond_dim, hidden_dim, latent_dim, out_channels, heads=1):
        super().__init__()
        self.encoder = ConditionalTransformerEncoder(in_channels, cond_dim, hidden_dim, latent_dim, heads)
        self.decoder = ConditionalTransformerDecoder(latent_dim, cond_dim, hidden_dim, out_channels, heads)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        cond = data.cond
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch, cond)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, data.edge_index, data.batch, cond)
        return recon_x, mu, logvar


#initialise the model

model = ConditionalGraphVAE(
    in_channels=4,       # xyz with atom element
    #updated to 221 features
    cond_dim=221,
    hidden_dim=32,
    latent_dim=32,
    out_channels=3
).to(device)

#fixed acetone geometry to use for hydrogen bond loss 
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

def water_only_loss(pred, target):
    return F.mse_loss(pred, target)

#functions to compute the bond lengths and angles losses


def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

def bond_angle(atom1, atom2, atom3):
    vec1 = atom1 - atom2
    vec2 = atom3 - atom2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure it's in range
    return np.degrees(angle)  # Convert to degrees

def compute_angle_loss(recon_x, target_angle=104.5):
    
    O = recon_x[:, 0, :]  #oxygen
    H1 = recon_x[:, 1, :]  # hydrogen 1
    H2 = recon_x[:, 2, :]  # hydrogen 2

    vec1 = H1 - O
    vec2 = H2 - O

    cos_angle = F.cosine_similarity(vec1, vec2, dim=1)
    angles = torch.acos(torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / np.pi)

    target_tensor = torch.full_like(angles, target_angle)
    angle_loss = F.mse_loss(angles, target_tensor)

    return angle_loss

#constrain the hydrogen bond distance as a loss function, early on in this study the water molecule was a long way from acetone

def compute_hbond_distance_range_loss(recon_x, min_dist=2.0, max_dist=4.0):

    O_water = recon_x[:, 0, :]  # water oxygen
    dist = torch.norm(O_water, dim=1)  # distance from origin -> oxygen on acetone

    too_close = F.relu(min_dist - dist)  
    too_far = F.relu(dist - max_dist)   

    return (too_close ** 2 + too_far ** 2).mean()


#training loop

import math
import torch.nn.functional as F


num_epochs = 1000
bond_loss_weight = 500
angle_weight = 100
condition_weight = 5
hbond_distance_weight = 100 
bond_target = 0.96  # ~O–H bond length


optimizer = optim.Adam(model.parameters(), lr=0.001)
condition_feature_weights = torch.ones(221).to(device)

#tracking losses
train_losses, val_losses = [], []
angle_losses, bond_errors, angle_errors = [], [], []
recon_losses, kl_losses, condition_losses = [], [], []
hbond_distance_losses = []

for epoch in range(num_epochs):
    # KL warm up 
    if epoch < 50:
        kl_weight = 0.0
    else:
        base = (epoch - 50) / (200 - 50)
        kl_weight = 0.01 * base * (0.5 * (1 + math.sin(epoch / 10)))

    model.train()
    epoch_train_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs

        optimizer.zero_grad()

        recon_x, mu, logvar = model(batch)
        target_x = batch.y.view(batch_size, 3, 3)

        recon_loss = F.mse_loss(recon_x, target_x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        O, H1, H2 = recon_x[:, 0, :], recon_x[:, 1, :], recon_x[:, 2, :]
        bond1 = torch.norm(O - H1, dim=1)
        bond2 = torch.norm(O - H2, dim=1)
        bond_target_tensor = torch.full_like(bond1, bond_target)
        bond_loss = F.mse_loss(bond1, bond_target_tensor) + F.mse_loss(bond2, bond_target_tensor)

        angle_loss = compute_angle_loss(recon_x)
        hbond_distance_loss = compute_hbond_distance_range_loss(recon_x, min_dist=2.0, max_dist=4.0)

        experimental_diff = batch.cond
        condition_target = torch.zeros_like(experimental_diff)
        condition_loss = F.mse_loss(experimental_diff * condition_feature_weights, condition_target)

        # combined loss 
        loss = (
            recon_loss +
            kl_weight * kl_loss +
            bond_loss_weight * bond_loss +
            angle_weight * angle_loss +
            hbond_distance_weight * hbond_distance_loss +
            condition_weight * condition_loss
        )


	#note if issues with the training 
        if not torch.isfinite(loss) or loss.item() > 1e6:
            print(f"Skipping unstable batch at epoch {epoch}: loss = {loss.item():.2e}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    # logging training epochs
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    recon_losses.append(recon_loss.item())
    kl_losses.append(kl_loss.item())
    condition_losses.append(condition_loss.item())
    hbond_distance_losses.append(hbond_distance_loss.item())

    #validation 
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs

            recon_x, mu, logvar = model(batch)
            recon_x = recon_x.view(batch_size, 3, 3)
            target_x = batch.y.view(batch_size, 3, 3)

            recon_loss = F.mse_loss(recon_x, target_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            O, H1, H2 = recon_x[:, 0, :], recon_x[:, 1, :], recon_x[:, 2, :]
            bond1 = torch.norm(O - H1, dim=1)
            bond2 = torch.norm(O - H2, dim=1)
            bond_target_tensor = torch.full_like(bond1, bond_target)
            bond_loss = F.mse_loss(bond1, bond_target_tensor) + F.mse_loss(bond2, bond_target_tensor)

            angle_loss = compute_angle_loss(recon_x)
            val_loss = (
                recon_loss +
                kl_weight * kl_loss +
                bond_loss_weight * bond_loss +
                angle_weight * angle_loss
            )

            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # evaluation
    model.eval()
    sample_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        batch_size = sample_batch.num_graphs
        recon_x, _, _ = model(sample_batch)
        recon_x = recon_x.view(batch_size, 3, 3)

        O, H1, H2 = recon_x[:, 0, :], recon_x[:, 1, :], recon_x[:, 2, :]
        bond1 = torch.norm(O - H1, dim=1)
        bond2 = torch.norm(O - H2, dim=1)
        mean_bond_err = ((bond1 - bond_target).abs().mean() + (bond2 - bond_target).abs().mean()) / 2
        bond_errors.append(mean_bond_err.item())

        O_np, H1_np, H2_np = O.cpu().numpy(), H1.cpu().numpy(), H2.cpu().numpy()
        angles = [bond_angle(H1_np[i], O_np[i], H2_np[i]) for i in range(batch_size)]
        mean_angle_error = np.abs(np.mean(angles) - 104.5)
        angle_errors.append(mean_angle_error)

    angle_losses.append(angle_loss.item())

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Bond Error: {mean_bond_err:.4f} Å | "
        f"Angle Error: {mean_angle_error:.2f}° | "
        f"H-bond Dist Loss: {hbond_distance_loss.item():.4f}"
    )



#sampling both deterministic and stochiastic outputs from the latent space

sample_batch = next(iter(train_loader)).to(device)
sample = sample_batch.to_data_list()[0]  # Take the first graph
sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=sample.x.device)

bond_lengths_OH1_mu = []
bond_lengths_OH2_mu = []
bond_angles_mu = []
oo_distances_mu = []

bond_lengths_OH1_sample = []
bond_lengths_OH2_sample = []
bond_angles_sample = []
oo_distances_sample = []

mu_geometries = []
sampled_geometries = []



for i in range(9):
    condition = torch.zeros((1, 221), device=device)
    condition[0, -1] = i / 9  # normalised concentration??

    with torch.no_grad():
        mu, logvar = model.encoder(sample.x, sample.edge_index, sample.batch, condition)

        # deterministic modelling 
        recon_mu = model.decoder(mu, sample.edge_index, sample.batch, condition)
        geom_mu = recon_mu.squeeze(0).cpu().numpy()
        mu_geometries.append(geom_mu)  # saving the output geometries 
        O_mu, H1_mu, H2_mu = geom_mu
        b1_mu = bond_length(O_mu, H1_mu)
        b2_mu = bond_length(O_mu, H2_mu)
        angle_mu = bond_angle(H1_mu, O_mu, H2_mu)
        dist_OO_mu = np.linalg.norm(O_mu)

        bond_lengths_OH1_mu.append(b1_mu)
        bond_lengths_OH2_mu.append(b2_mu)
        bond_angles_mu.append(angle_mu)
        oo_distances_mu.append(dist_OO_mu)

        # sampling
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_sample = model.decoder(z, sample.edge_index, sample.batch, condition)
        geom_sample = recon_sample.squeeze(0).cpu().numpy()
        sampled_geometries.append(geom_sample)  # saving the output geometries
        O_s, H1_s, H2_s = geom_sample
        b1_s = bond_length(O_s, H1_s)
        b2_s = bond_length(O_s, H2_s)
        angle_s = bond_angle(H1_s, O_s, H2_s)
        dist_OO_s = np.linalg.norm(O_s)

        bond_lengths_OH1_sample.append(b1_s)
        bond_lengths_OH2_sample.append(b2_s)
        bond_angles_sample.append(angle_s)
        oo_distances_sample.append(dist_OO_s)

        # === Print comparison ===
        print(f"Concentration {i}/9:")
        print(f"? mu   : O–H1 {b1_mu:.4f} Å, O–H2 {b2_mu:.4f} Å, angle {angle_mu:.2f}°, O–O {dist_OO_mu:.4f} Å")
        print(f"? sample: O–H1 {b1_s:.4f} Å, O–H2 {b2_s:.4f} Å, angle {angle_s:.2f}°, O–O {dist_OO_s:.4f} Å")
        print("=====================================")



#plotting results  
import matplotlib.pyplot as plt

concentrations = list(range(9))

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.ravel()

#plot O–H1 bond length
axs[0].plot(concentrations, bond_lengths_OH1_mu, label='µ (mean)', marker='o')
axs[0].plot(concentrations, bond_lengths_OH1_sample, label='sampled', marker='x')
axs[0].set_title('O–H1 Bond Length')
axs[0].set_ylabel('Å')
axs[0].legend()

#plot O–H2 bond length
axs[1].plot(concentrations, bond_lengths_OH2_mu, label='µ (mean)', marker='o')
axs[1].plot(concentrations, bond_lengths_OH2_sample, label='sampled', marker='x')
axs[1].set_title('O–H2 Bond Length')
axs[1].set_ylabel('Å')
axs[1].legend()

#plot H–O–H angle
axs[2].plot(concentrations, bond_angles_mu, label='µ (mean)', marker='o')
axs[2].plot(concentrations, bond_angles_sample, label='sampled', marker='x')
axs[2].set_title('H–O–H Angle')
axs[2].set_ylabel('Degrees')
axs[2].legend()

#plot O–O distance
axs[3].plot(concentrations, oo_distances_mu, label='µ (mean)', marker='o')
axs[3].plot(concentrations, oo_distances_sample, label='sampled', marker='x')
axs[3].set_title('O (acetone) – O (water) Distance')
axs[3].set_ylabel('Å')
axs[3].legend()

for ax in axs:
    ax.set_xlabel('Concentration index')

plt.tight_layout()
plt.show()

#plot geometries 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#atom labels and bonding for plotting
labels = ['O', 'H1', 'H2']

acetone_coords = acetone_template[:, 1:].cpu().numpy()
acetone_bonds = [
    (0, 1),  # O–C
    (1, 2), (1, 3),  # C–C
    (2, 4), (2, 5), (2, 6),  # C–H
    (3, 7), (3, 8), (3, 9)   # C–H
]

fig = plt.figure(figsize=(18, 10))
labels = ['O', 'H1', 'H2']

for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, projection='3d')
    mu = mu_geometries[i]
    sample = sampled_geometries[i]

    #plot acetone
    ax.scatter(acetone_coords[:, 0], acetone_coords[:, 1], acetone_coords[:, 2], c='gray', label='Acetone', s=20)
    for a, b in acetone_bonds:
        ax.plot(
            [acetone_coords[a, 0], acetone_coords[b, 0]],
            [acetone_coords[a, 1], acetone_coords[b, 1]],
            [acetone_coords[a, 2], acetone_coords[b, 2]],
            c='black', linewidth=1
        )

    #plot water µ results
    ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c='blue', label='µ (mean)', s=50)
    for j in range(3):
        ax.text(mu[j, 0], mu[j, 1], mu[j, 2], labels[j], color='blue')
    for H in [mu[1], mu[2]]:
        ax.plot([mu[0, 0], H[0]], [mu[0, 1], H[1]], [mu[0, 2], H[2]], c='blue', linewidth=1)

    #plot sampled water results
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='red', label='sample', marker='^', s=50)
    for j in range(3):
        ax.text(sample[j, 0], sample[j, 1], sample[j, 2], labels[j], color='red')
    for H in [sample[1], sample[2]]:
        ax.plot([sample[0, 0], H[0]], [sample[0, 1], H[1]], [sample[0, 2], H[2]], c='red', linewidth=1)

    ax.set_title(f'Concentration {i}/9')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.axis('off')

plt.tight_layout()
plt.show()
