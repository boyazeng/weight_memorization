import numpy as np
import os
import sys
import torch

from Gpt.tasks import TASK_METADATA
from Gpt.data.normalization import get_normalizer
from Gpt.vis import moduleify

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_min_distance_to_training import draw
from permute_util import calculate_min_distance_and_best_permutation

if __name__ == "__main__":
    full_training_weights = torch.load("data/training_weights.pt")
    full_generated_weights = torch.load("data/generated_weights.pt")
    normalizer = get_normalizer("openai", openai_coeff=4.185)
    training_ckpts = moduleify(full_training_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)
    generated_ckpts = moduleify(full_generated_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)

    if os.path.exists("data/min_distances_training.npy"):
        min_distances_training = np.load("data/min_distances_training.npy")
        min_distances_generated = np.load("data/min_distances_generated.npy")
    else:
        min_distances_training, _ = calculate_min_distance_and_best_permutation(training_ckpts, training_ckpts)
        min_distances_generated, _ = calculate_min_distance_and_best_permutation(training_ckpts, generated_ckpts)
        min_distances_training = np.min(min_distances_training, axis=0)
        min_distances_generated = np.min(min_distances_generated, axis=0)
        np.save("data/min_distances_training.npy", min_distances_training)
        np.save("data/min_distances_generated.npy", min_distances_generated)
    draw(min_distances_training, min_distances_generated, os.path.abspath("figures"))