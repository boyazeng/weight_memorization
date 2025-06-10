import os; os.chdir("../method")
import sys; sys.path.append("../method")
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KernelDensity
from src.ghrp.model_definitions.def_simclr_ae_module import SimCLRAEModule
import torch

def flatten_checkpoint(checkpoint):
    flattened_values = [value.flatten() for value in checkpoint.values()]
    return torch.cat(flattened_values)

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    experiment_path = Path("./data/hyper_representations/svhn")

    dataset = torch.load(experiment_path / "dataset.pt", weights_only=False)
    weights_train = dataset["trainset"].__get_weights__()  # shape [2896, 2464]

    checkpoint = torch.load(experiment_path.joinpath(f"checkpoint_ae.pt"), map_location="cpu", weights_only=False)
    config = json.load((experiment_path / "config_ae.json").open("r"))
    config.update({"device": "cpu", "model::type": "transformer"})
    AE = SimCLRAEModule(config)
    AE.model.load_state_dict(checkpoint)
    AE.model.eval()

    with torch.no_grad():
        z_train = AE.forward_encoder(weights_train)
    train_zoo_acc = torch.Tensor(dataset["trainset"].properties["test_acc"])
    acc_threshold = torch.quantile(train_zoo_acc, q=0.7)
    idx_best = [
        idx
        for idx, acc_dx in enumerate(dataset["trainset"].properties["test_acc"])
        if acc_dx >= acc_threshold
    ]

    kde = KernelDensity(kernel="gaussian", bandwidth=0.002)
    z_samples_top30 = []
    for dim in range(z_train.shape[1]):
        kde.fit(z_train[idx_best, dim].unsqueeze(1))
        z_top30_tmp = kde.sample(n_samples=200, random_state=42)
        z_samples_top30.append(torch.tensor(z_top30_tmp))
    z_samples_top30 = torch.cat(z_samples_top30, dim=1).float()
    with torch.no_grad():
        decoded_weights_z_samples_top30 = AE.forward_decoder(z_samples_top30)
    np.save("../data/generated_weights.npy", decoded_weights_z_samples_top30)