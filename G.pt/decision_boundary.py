import numpy as np
import os
import sys
import torch

from data_gen.train_mnist import unload_test_set
from Gpt.tasks import TASK_METADATA
from Gpt.data.normalization import get_normalizer
from Gpt.vis import moduleify
from permute_util import calculate_min_distance_and_best_permutation, permute

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_decision_boundary import draw, generate_pca_grid

def mask_off_diagonal(matrix):
    diag_mask = np.eye(*matrix.shape, dtype=bool)
    return np.where(diag_mask, matrix, np.nan)

@torch.no_grad()
def main():
    full_training_weights = torch.load("data/training_weights.pt")
    full_generated_weights = torch.load("data/generated_weights.pt")[:3]
    normalizer = get_normalizer("openai", openai_coeff=4.185)
    training_ckpts = moduleify(full_training_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)
    generated_ckpts = moduleify(full_generated_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)

    min_dist_train_gen, best_training_permutation = calculate_min_distance_and_best_permutation(training_ckpts, generated_ckpts)
    nearest_training_ckpt_indices = np.argmin(min_dist_train_gen, axis=0)
    permutation_nearest_training = best_training_permutation[nearest_training_ckpt_indices, np.arange(3)] # shape = [num_generated=3, num_hidden_neuron=10]

    images = unload_test_set()[0]
    grid_images_tensor = generate_pca_grid(images)

    grid_preds = [[] for _ in range(3)]
    for generated_id in range(3):
        nearest_training_ckpt = permute(training_ckpts[nearest_training_ckpt_indices[generated_id]], permutation_nearest_training[generated_id])
        generated_ckpt = generated_ckpts[generated_id].to("cuda")
        grid_preds[generated_id].append(generated_ckpt(grid_images_tensor.cuda()).argmax(dim=1).cpu().numpy())
        grid_preds[generated_id].append(nearest_training_ckpt(grid_images_tensor.cuda()).argmax(dim=1).cpu().numpy())

    draw(grid_preds, 300, "figures")

if __name__ == "__main__":
    main()