

#VAE


!pip install torch_geometric

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim

#define the device as cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#use the cpu for testing as cuda hides the issue if its difference in vector lengths etc
#device=('cpu')
torch.autograd.set_detect_anomaly(True)



#load the dataset
dataset=torch.load("graph_dataset_with_H8.pt",weights_only=False)
print(f";oaded dataset with {len(dataset)} graphs")



#verify dataset attributes
print(f"graph Keys: {dataset[0].keys()}")  # Should include 'x', 'edge_index', 'qm_features', 'y'

#define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ensure splits sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "no"

train_data, temp_data = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

from torch_geometric.loader import DataLoader

#create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#describe the decoder

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ConditionalGNNDecoder(torch.nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, out_channels):
        super().__init__()
        self.fc_cond = torch.nn.Linear(cond_dim, latent_dim)  # map condition to latent space
        self.conv1 = GCNConv(latent_dim, hidden_dim)          # graph Convolution Layer 1
        self.conv2 = GCNConv(hidden_dim, out_channels)        # graph Convolution Layer 2

        # Acetone reference geometry (atom 0–9)
        self.acetone_coords = torch.tensor([
            [8,  0.00000,  0.00000,  0.00000],   # O1
            [6, -0.18923,  0.84171,  0.86495],   # C1
            [6,  0.96812,  1.38841,  1.64288],   # C2
            [6, -1.57836,  1.32603,  1.14647],   # C3
            [1,  1.03349,  2.48631,  1.49265],   # H1
            [1,  0.82576,  1.17534,  2.72296],   # H2
            [1,  1.92064,  0.92411,  1.30898],   # H3
            [1, -2.31283,  0.82041,  0.48370],   # H4
            [1, -1.63426,  2.42096,  0.97259],   # H5
            [1, -1.84199,  1.11000,  2.20291],   # H6
        ], dtype=torch.float)  # shape: [10, 4]

    def forward(self, z, edge_index, batch, condition):
        cond_embed = self.fc_cond(condition)
        cond_expanded = cond_embed[batch]
        z = z + cond_expanded

        x = F.relu(self.conv1(z, edge_index))
        x = self.conv2(x, edge_index)

        # reshape to [batch_size, 13, 4]
        num_graphs = batch.max().item() + 1
        x = x.view(num_graphs, 13, 4)

        # fix acetone atoms (0–9)
        fixed_acetone = self.acetone_coords.to(x.device).unsqueeze(0).expand(num_graphs, -1, -1)
        x[:, 0:10, :] = fixed_acetone

        # displacement to carbonyl oxygen position (atom 0)
        carbonyl_O = fixed_acetone[:, 0, 1:].unsqueeze(1)  # shape: [batch, 1, 3]
        displacement = x[:, 10:13, 1:]                     # shape: [batch, 3, 3]
        x[:, 10:13, 1:] = carbonyl_O + displacement        # water = atom 10–12

        return x

#describe the encoder

class ConditionalGNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.fc_cond = torch.nn.Linear(cond_dim, latent_dim)
        self.mu_layer = GCNConv(latent_dim, latent_dim)
        self.logvar_layer = GCNConv(latent_dim, latent_dim)

    def forward(self, x, edge_index, batch, condition):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        cond_embed = self.fc_cond(condition)
        cond_expanded = cond_embed[batch]

        x = x + cond_expanded

        #clamp x before going into logvar layer
        x = torch.clamp(x, -10, 10)  # Avoid exploding values

        mu = self.mu_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        logvar = torch.clamp(logvar, min=-5, max=5)

        return mu, logvar

#describe the VAE

class ConditionalGraphVAE(torch.nn.Module):
    def __init__(self, in_channels, cond_dim, hidden_dim, latent_dim, out_channels):
        super().__init__()
        self.encoder = ConditionalGNNEncoder(in_channels, cond_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalGNNDecoder(latent_dim, cond_dim, hidden_dim, out_channels)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, data):
        cond = data.y.view(-1, 10)
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch, cond)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, data.edge_index, data.batch, cond)
        return recon_x, mu, logvar

        print("mu stats:", mu.min().item(), mu.max().item())
        print("logvar stats:", logvar.min().item(), logvar.max().item())

model = ConditionalGraphVAE(
    in_channels=4,       # atomic num + xyz
    cond_dim=10,         # IR shift target
    hidden_dim=64,
    latent_dim=32,
    out_channels=4
).to(device)

# fixed acetone geometry for atoms 0–9 (shape: [10, 4])
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

def water_only_loss(pred, target, num_graphs):
    # Water atom indices per graph (atoms 10, 11, 12)
    water_idx = torch.tensor([10, 11, 12], device=pred.device)
    mask = torch.cat([
        water_idx + i * 13 for i in range(num_graphs)
    ])
    return F.mse_loss(pred[mask], target[mask])

def fix_acetone_atoms(recon_x, acetone_template):
    """
    Overwrites atoms 0–9 (acetone) in each graph in a batch with a fixed geometry,
    without modifying water atoms or causing in-place operations that break autograd.
    """
    batch_size = recon_x.shape[0]
    acetone_repeated = acetone_template.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 10, 4]

    # combine acetone with predicted water (10:13)
    water = recon_x[:, 10:13, :].clone()  # detach water part safely
    fixed = torch.cat([acetone_repeated, water], dim=1)
    return fixed

#define bond angles and lengths

# function to compute bond length
def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

# function to compute bond angle
def bond_angle(atom1, atom2, atom3):
    vec1 = atom1 - atom2
    vec2 = atom3 - atom2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure it's in range
    return np.degrees(angle)  # Convert to degrees

def compute_angle_loss(recon_x, target_angle=104.5):
    O = recon_x[:, 10, 1:]
    H1 = recon_x[:, 11, 1:]
    H2 = recon_x[:, 12, 1:]

    vec1 = H1 - O
    vec2 = H2 - O
    cos_angle = F.cosine_similarity(vec1, vec2, dim=1)
    angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)) * (180.0 / np.pi)

    angle_loss = F.mse_loss(angles, torch.full_like(angles, target_angle))
    return angle_loss

# train the model with a gradual KL divergence weight increase
optimizer = optim.Adam(model.parameters(), lr=0.0005 )  # Lower learning rate for better convergence

num_epochs = 2000  # Increase epochs for better training
train_losses = []
val_losses = []
angle_losses=[]
bond_errors = []  # Track mean O–H bond error over time
angle_errors = []  # Track mean H1–O–H2 bond angle error over time
recon_losses = []
kl_losses = []

#adjust this to improve the performance of the model
bond_loss_weight = 1
angle_weight = 10


bond_target = 0.96  # expected bond length in Å

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    # increase KL weight over epochs
    kl_weight = min(1.0, epoch / 200)  # Slow down KL growth

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(batch)
        batch_size = batch.num_graphs
        recon_x = recon_x.view(batch_size, 13, 4)
        target_x = batch.x.view(batch_size, 13, 4)

       
        target_water_before = target_x[:, 10:13, :].clone()

        #replace acetone atoms (0–9) while preserving water atoms (10–12)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)
        target_x = fix_acetone_atoms(target_x.clone(), acetone_template)

        #check water atoms weren't changed
        assert torch.allclose(target_water_before, target_x[:, 10:13, :]), "🚨 Water atoms modified!"


        recon_loss = F.mse_loss(recon_x[:, 10:13, :], target_x[:, 10:13, :])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        #bond length penalty
        O = recon_x[:, 10, 1:]
        H1 = recon_x[:, 11, 1:]
        H2 = recon_x[:, 12, 1:]
        bond1 = torch.norm(O - H1, dim=1)
        bond2 = torch.norm(O - H2, dim=1)
        bond_target_tensor = torch.full_like(bond1, bond_target)
        bond_loss = F.mse_loss(bond1, bond_target_tensor) + F.mse_loss(bond2, bond_target_tensor)
        angle_loss = compute_angle_loss(recon_x)




        # final loss with KL weight warm-up
        loss = recon_loss + kl_weight * kl_loss + bond_loss_weight * bond_loss + angle_weight * angle_loss



        #skip this batch if loss is non-finite or huge
        if not torch.isfinite(loss) or loss.item() > 1e6:
            print(f"skip unstable batch at epoch {epoch}: loss = {loss.item():.2e}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    recon_losses.append(recon_loss.item())
    kl_losses.append(kl_loss.item())


    #validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon_x, mu, logvar = model(batch)
            batch_size = batch.num_graphs
            recon_x = recon_x.view(batch_size, 13, 4)
            target_x = batch.x.view(batch_size, 13, 4)
            recon_x = fix_acetone_atoms(recon_x, acetone_template)

            recon_loss = F.mse_loss(recon_x[:, 10:13, :], target_x[:, 10:13, :])
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            #bond penalty
            O = recon_x[:, 10, 1:]
            H1 = recon_x[:, 11, 1:]
            H2 = recon_x[:, 12, 1:]
            bond1 = torch.norm(O - H1, dim=1)
            bond2 = torch.norm(O - H2, dim=1)
            bond_target_tensor = torch.full_like(bond1, bond_target)
            bond_loss = F.mse_loss(bond1, bond_target_tensor) + F.mse_loss(bond2, bond_target_tensor)

            #final loss with KL weight warm-up
            angle_loss = compute_angle_loss(recon_x)
            loss = recon_loss + kl_weight * kl_loss + bond_loss_weight * bond_loss + angle_weight * angle_loss

            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)



    # bond error tracking on a sample batch
    model.eval()
    # bond length and angle error tracking on a sample batch
    sample_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        recon_x, _, _ = model(sample_batch)
        batch_size = recon_x.shape[0]
        recon_x = recon_x.view(batch_size, 13, 4)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)

        #bond lengths
        O = recon_x[:, 10, 1:]
        H1 = recon_x[:, 11, 1:]
        H2 = recon_x[:, 12, 1:]
        bond1 = torch.norm(O - H1, dim=1)
        bond2 = torch.norm(O - H2, dim=1)
        mean_bond_err = ((bond1 - bond_target).abs().mean() + (bond2 - bond_target).abs().mean()) / 2
        bond_errors.append(mean_bond_err.item())

        #bond angles
        O_np = O.cpu().numpy()
        H1_np = H1.cpu().numpy()
        H2_np = H2.cpu().numpy()
        angles = []
        for i in range(batch_size):
            angle = bond_angle(H1_np[i], O_np[i], H2_np[i])  # angle in degrees
            angles.append(angle)
        mean_angle_error = np.abs(np.mean(angles) - 104.5)
        angle_errors.append(mean_angle_error)




    angle_losses.append(angle_loss.item())  #track the angle_loss

    print(f"epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Mean O–H Bond Error: {mean_bond_err:.4f} Å  | Mean angle error : {mean_angle_error:.4f}")



torch.save(model.state_dict(), "1000.pt")

import matplotlib.pyplot as plt

plt.plot(bond_errors)
plt.xlabel("epoch")
plt.ylabel("mean O–H Bond Error (Å)")
plt.title("bond Length Learning Over Time")
plt.grid(True)
plt.show()

plt.plot(angle_errors)
plt.xlabel("epoch")
plt.ylabel("mean H–O–H angle error (°)")
plt.title("water bond angle error over training")
plt.grid(True)
plt.show()

plt.plot(train_losses, label="train Loss")
plt.plot(val_losses, label="validation Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training vs validation Loss")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(recon_losses, label='reconstruction loss')
plt.plot(kl_losses, label='KL divergence loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('reconstruction vs KL loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#predictions
from torch_geometric.loader import DataLoader

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model.eval()
test_loss = 0
recon_losses = []
kl_losses = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        recon_x, mu, logvar = model(batch)
        batch_size = batch.num_graphs
        recon_x = recon_x.view(batch_size, 13, 4)
        target_x = batch.x.view(batch_size, 13, 4)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)

        recon_loss = F.mse_loss(recon_x[:, 10:13, :], target_x[:, 10:13, :])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        test_loss += loss.item()
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())

# average test losses
avg_test_loss = test_loss / len(test_loader)
avg_recon = sum(recon_losses) / len(recon_losses)
avg_kl = sum(kl_losses) / len(kl_losses)

print(f"test loss {avg_test_loss:.4f}")
print(f"reconstruction loss: {avg_recon:.4f}")
print(f"KL divergence loss: {avg_kl:.4f}")

#function to compute bond length, make sure its in angstroms!
def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)*10

#function to compute bond angle
def bond_angle(atom1, atom2, atom3):
    vec1 = atom1 - atom2
    vec2 = atom3 - atom2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure it's in range
    return np.degrees(angle)*1000  # Convert to degrees

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
plt.ylim(0, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

# plot bond angle
plt.figure(figsize=(6, 4))
plt.plot(bond_angles, marker='o', label='H–O–H Angle')
plt.axhline(104.5, color='gray', linestyle='--', label='Target')
plt.title("Water Bond Angle vs Concentration")
plt.xlabel("Concentration Index")
plt.ylabel("Angle (°)")
plt.legend()
plt.ylim(0, 200)
plt.grid(True)
plt.tight_layout()
plt.show()