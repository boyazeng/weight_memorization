import numpy as np
import os
import sys
import torch

from Gpt.tasks import TASK_METADATA
from Gpt.data.normalization import get_normalizer
from Gpt.vis import moduleify
from permute_util import calculate_min_distance_and_best_permutation, permute

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_heatmap import draw

def flatten(mod):
    return torch.cat([mod.fc1.weight.flatten(), mod.fc1.bias.flatten(), mod.fc2.weight.flatten(), mod.fc2.bias.flatten()])

@torch.no_grad()
def main():
    training_weights = torch.load("data/training_weights.pt")
    generated_weights = torch.load("data/generated_weights.pt")[:3]
    normalizer = get_normalizer("openai", openai_coeff=4.185)
    training_ckpts = moduleify(training_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)
    generated_ckpts = moduleify(generated_weights, TASK_METADATA["mnist_loss"]["constructor"], normalizer.unnormalize)

    min_dist_train_gen, best_training_permutation = calculate_min_distance_and_best_permutation(training_ckpts, generated_ckpts)
    nearest_training_ckpt_indices = np.argsort(min_dist_train_gen, axis=0)[:3] # shape = [num_nearest_training=3, num_generated=3]
    permutation_nearest_training = [best_training_permutation[nearest_training_ckpt_indices[i], np.arange(3)] for i in range(3)] # shape = [num_nearest_training=3, num_generated=3, num_hidden_neuron=10]
    param_indices = sorted(np.random.choice(flatten(generated_ckpts[0].to("cuda")).shape[0], size=64, replace=False))

    generated, nearest_training = [], []
    for generated_id in range(3):
        nearest_training_ckpts = [permute(training_ckpts[nearest_training_ckpt_indices[i][generated_id]], permutation_nearest_training[i][generated_id]) for i in range(3)]
        generated_ckpt = generated_ckpts[generated_id].to("cuda")
        generated_weight, nearest_training_weights = flatten(generated_ckpt)[param_indices], [flatten(it)[param_indices] for it in nearest_training_ckpts]
        generated.append(generated_weight.cpu().numpy())
        nearest_training.append([it.cpu().numpy() for it in nearest_training_weights])
    draw(generated, nearest_training, os.path.abspath("figures"))

if __name__ == "__main__":
    main()