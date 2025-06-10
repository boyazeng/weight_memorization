import numpy as np
import os
from pathlib import Path
import sys
import torch

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_min_distance_to_training import calculate_min_distances, draw

if __name__ == "__main__":
    experiment_path = Path("./method/data/hyper_representations/svhn")
    reconstructed_weights = torch.tensor(np.load("./data/reconstructed_weights.npy"), device="cuda")
    gen_weights = torch.tensor(np.load("./data/generated_weights.npy"), device="cuda")
    epochs = torch.load(experiment_path / "dataset.pt", weights_only=False)["trainset"].epochs
    epochs = torch.tensor(epochs)
    reconstructed_weights = reconstructed_weights[epochs == 25]

    min_distances_reconstructed = calculate_min_distances(reconstructed_weights, reconstructed_weights)
    min_distances_generated = calculate_min_distances(reconstructed_weights, gen_weights)
    draw(min_distances_reconstructed, min_distances_generated, os.path.abspath("figures"))