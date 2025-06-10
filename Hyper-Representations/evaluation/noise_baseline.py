import os; os.chdir("../method")
import sys; sys.path.append("../method")
import numpy as np
from pathlib import Path
import torch

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    experiment_path = Path("./data/hyper_representations/svhn")
    ckpt_trainset = torch.load(experiment_path / "dataset.pt", weights_only=False)["trainset"]
    training_weights = torch.tensor(np.load("../data/reconstructed_weights.npy"))

    train_zoo_acc = torch.Tensor(ckpt_trainset.properties["test_acc"])
    acc_threshold = torch.quantile(train_zoo_acc, q=0.7)
    idx_best = [
        idx
        for idx, acc_dx in enumerate(ckpt_trainset.properties["test_acc"])
        if acc_dx >= acc_threshold
    ]
    training_weights = training_weights[idx_best]

    for noise_level in [0.02, 0.04]:
        weight_vector = training_weights + torch.randn_like(training_weights) * noise_level
        np.save(f"../data/noise{noise_level}_weights.npy", weight_vector)