import numpy as np
import os
from pathlib import Path
import sys
import torch
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_scatter_plot import draw, max_similarity_to_training, randomly_select_100

if __name__ == "__main__":
    experiment_path = Path("./method/data/hyper_representations/svhn")
    training_epochs = np.array(torch.load(experiment_path / "dataset.pt", weights_only=False)["trainset"].epochs)

    training_incorrect_indices = np.load("./data/reconstructed_incorrect_indices.npy")
    novelty = []
    performance = []
    types = ["training", "generated", "noise0.02", "noise0.04"]
    for type in types:
        if type == "training":
            incorrect_indices = np.load(f"./data/reconstructed_incorrect_indices.npy")
            accs = np.load(f"./data/reconstructed_accs.npy") * 100
        else:
            incorrect_indices = np.load(f"./data/{type}_incorrect_indices.npy")
            accs = np.load(f"./data/{type}_accs.npy") * 100
        similarity = max_similarity_to_training(training_incorrect_indices, incorrect_indices, batch_size=32)
        similarity, accs = randomly_select_100(similarity, accs)
        novelty.append(similarity)
        performance.append(accs)
    draw(novelty, performance, "maximum similarity", "accuracy (%)", types, os.path.abspath("figures/performance_vs_novelty.png"))