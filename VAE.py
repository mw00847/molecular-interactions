#this script runs a conditional graph neural network and took around 3-4 hours to run 50,0000 epochs on a NVIDIA GeForce RTX 4060 with 8GB

#import modules 

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

#load dataset 

dataset=torch.load("graph_dataset_with_H8.pt",weights_only=False)
print(f"loaded dataset with {len(dataset)} graphs")

#split the data into train and validation

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# train/Val/test Split
train_data, temp_data = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

from torch_geometric.loader import DataLoader

# create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#describe the decoder

class ConditionalGNNDecoder(torch.nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, out_channels):
        super().__init__()
        self.fc_cond = torch.nn.Linear(cond_dim, latent_dim)  
        self.conv1 = GCNConv(latent_dim, hidden_dim)         
        self.conv2 = GCNConv(hidden_dim, out_channels)        

        #acetone reference geometry (atom 0–9)
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

        #reshape to [batch_size, 13, 4]
        num_graphs = batch.max().item() + 1
        x = x.view(num_graphs, 13, 4)

        #fix acetone atoms (0–9)
        fixed_acetone = self.acetone_coords.to(x.device).unsqueeze(0).expand(num_graphs, -1, -1)
        x[:, 0:10, :] = fixed_acetone

        #add displacement to carbonyl oxygen position (atom 0)
        carbonyl_O = fixed_acetone[:, 0, 1:].unsqueeze(1)  # shape: [batch, 1, 3]
        displacement = x[:, 10:13, 1:]                     # shape: [batch, 3, 3]
        x[:, 10:13, 1:] = carbonyl_O + displacement        # water = atom 10–12
`
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

        # clamp x before going into logvar layer
        x = torch.clamp(x, -10, 10)  # Avoid exploding values

        mu = self.mu_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        logvar = torch.clamp(logvar, min=-5, max=5)  # 🚨 Clamp to avoid instability

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


if __name__ == "__main__":
    print(running VAE training directly (not imported)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #look at adjusting these properties to improve the model?
    model = ConditionalGraphVAE(
        in_channels=4,
        cond_dim=10,
        hidden_dim=64,
        latent_dim=32,
        out_channels=4
    ).to(device)


#fixed acetone geometry for atoms 0–9 (shape: [10, 4])
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

    batch_size = recon_x.shape[0]
    acetone_repeated = acetone_template.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 10, 4]

    # Combine acetone with predicted water (10:13)
    water = recon_x[:, 10:13, :].clone()  # detach water part safely
    fixed = torch.cat([acetone_repeated, water], dim=1)
    return fixed

#define bond angles and lengths

#compute bond length
def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

#compute bond angle
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

#train the model with a gradual KL divergence weight increase
optimizer = optim.Adam(model.parameters(), lr=0.0001 )  

num_epochs = 50000  
train_losses = []
val_losses = []
angle_losses=[]
bond_errors = []  
angle_errors = []  
recon_losses = []
kl_losses = []

bond_loss_weight = 1
angle_weight = 10

bond_target = 0.96  # expected bond length in Å

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    #repeat KL schedule every 5000 epochs
    cycle_epoch = epoch % 5000

    if cycle_epoch < 1250:
        kl_weight = 0.0
    elif 1250 <= cycle_epoch < 2500:
        kl_weight = (cycle_epoch - 1250) / 1250  # 0.0 → 1.0
    elif 2500 <= cycle_epoch < 3750:
        kl_weight = 1.0 - ((cycle_epoch - 2500) / 1250)  # 1.0 → 0.0
    else:
        kl_weight = 0.0


    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(batch)
        batch_size = batch.num_graphs
        recon_x = recon_x.view(batch_size, 13, 4)
        target_x = batch.x.view(batch_size, 13, 4)

        #check fix_acetone_atoms
        target_water_before = target_x[:, 10:13, :].clone()

        # replace acetone atoms (0–9) while preserving water atoms (10–12)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)
        target_x = fix_acetone_atoms(target_x.clone(), acetone_template)

        # double-check water atoms weren't changed
        assert torch.allclose(target_water_before, target_x[:, 10:13, :]), "Water atoms modified!"


	#?
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

        # skip batch if loss is non-finite or huge
        if not torch.isfinite(loss) or loss.item() > 1e6:
            print(f" Skipping unstable batch at epoch {epoch}: loss = {loss.item():.2e}")
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

            # bond penalty again
            O = recon_x[:, 10, 1:]
            H1 = recon_x[:, 11, 1:]
            H2 = recon_x[:, 12, 1:]
            bond1 = torch.norm(O - H1, dim=1)
            bond2 = torch.norm(O - H2, dim=1)
            bond_target_tensor = torch.full_like(bond1, bond_target)
            bond_loss = F.mse_loss(bond1, bond_target_tensor) + F.mse_loss(bond2, bond_target_tensor)

            # final loss with KL weight warm-up
            angle_loss = compute_angle_loss(recon_x)
            loss = recon_loss + kl_weight * kl_loss + bond_loss_weight * bond_loss + angle_weight * angle_loss

            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)



    # bond error tracking on a sample batch
    model.eval()
    # bond length and angle error tracking on sample batch
    sample_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        recon_x, _, _ = model(sample_batch)
        batch_size = recon_x.shape[0]
        recon_x = recon_x.view(batch_size, 13, 4)
        recon_x = fix_acetone_atoms(recon_x, acetone_template)

        # bond lengths
        O = recon_x[:, 10, 1:]
        H1 = recon_x[:, 11, 1:]
        H2 = recon_x[:, 12, 1:]
        bond1 = torch.norm(O - H1, dim=1)
        bond2 = torch.norm(O - H2, dim=1)
        mean_bond_err = ((bond1 - bond_target).abs().mean() + (bond2 - bond_target).abs().mean()) / 2
        bond_errors.append(mean_bond_err.item())

        # bond angles
        O_np = O.cpu().numpy()
        H1_np = H1.cpu().numpy()
        H2_np = H2.cpu().numpy()
        angles = []
        for i in range(batch_size):
            angle = bond_angle(H1_np[i], O_np[i], H2_np[i])  # angle in degrees
            angles.append(angle)
        mean_angle_error = np.abs(np.mean(angles) - 104.5)
        angle_errors.append(mean_angle_error)




    angle_losses.append(angle_loss.item())  

    print(f"epoch {epoch+1}/{num_epochs} | KL weight: {kl_weight:.4f} | train Loss: {avg_train_loss:.4f} |val Loss: {avg_val_loss:.4f} |mean O–H Bond Error: {mean_bond_err:.4f} Å  | mean angle error : {mean_angle_error:.4f}")

torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'bond_errors': bond_errors,
    'angle_errors': angle_errors,
    'recon_losses': recon_losses,
    'kl_losses': kl_losses
}, "50000_checkpoint.pt")


torch.save({
    'train_losses': train_losses,
    'val_losses': val_losses,
    'bond_errors': bond_errors,
    'angle_errors': angle_errors,
    'recon_losses': recon_losses,
    'kl_losses': kl_losses
}, "training_results_50000.pt")



