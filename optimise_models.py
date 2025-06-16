#change the GNNN model that is imported to use different models 

import time 
import torch
import optuna
import sys
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from train_runner import train_model


from graph_utils import GraphWithCond
sys.modules['__main__'].GraphWithCond = GraphWithCond

# import just the GCN model
from GCN_VAE import ConditionalGraphVAE as GNNModel

# load the dataset 
dataset = torch.load("graph_dataset_MAY_25.pt")
print(f"loaded dataset with {len(dataset)} graphs")

#split dataset
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, _ = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# objective function describing parameters 

def objective(trial):
    num_epochs = trial.suggest_int("num_epochs", 50, 200)
    bond_loss_weight = trial.suggest_float("bond_loss_weight", 1, 10000.0, log=True)
    angle_weight = trial.suggest_float("angle_weight", 1, 10000, log=True)
    kl_weight_max = trial.suggest_float("kl_weight_max", 0.001, 1)

    model = GNNModel(
        in_channels=3,
        cond_dim=217,
        hidden_dim=64,
        latent_dim=64,
        out_channels=3
    ).to(device)

    val_loss,bond_err,angle_err = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        bond_loss_weight=bond_loss_weight,
        angle_weight=angle_weight,
        kl_weight_max=kl_weight_max,
    )

    # store metrics
    trial.set_user_attr("bond_error", bond_err)
    trial.set_user_attr("angle_error", angle_err)



    return val_loss

#optimisation
if __name__ == "__main__":
    start_time=time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    elapsed=time.time()-start_time
    print("elapsed time", elapsed)

    print("\nb est trial:")
    print(f"  validation Loss: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


# save trial history
df = study.trials_dataframe()
df.to_csv("optuna_trials_transformer.csv", index=False)
print("saved all trial results to optuna_trials.csv")


