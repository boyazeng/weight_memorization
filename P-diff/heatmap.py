import numpy as np
import os
import sys
import torch
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_heatmap import draw

config = {
    "training_path": "./method/dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": "./method/dataset/main/cifar100_resnet18/generated",
}

def dict_to_flat(diction):
    return torch.cat([item.flatten() for item in diction.values()])

if __name__ == "__main__":
    training_weights = [torch.load(os.path.join(config["training_path"], path)) for path in sorted(os.listdir(config["training_path"]))]
    generated_weights = [torch.load(os.path.join(config["generated_path"], path)) for path in sorted(os.listdir(config["generated_path"]))]
    training_weights = torch.stack([dict_to_flat(item) for item in training_weights]).cpu().numpy()
    generated_weights = torch.stack([dict_to_flat(item) for item in generated_weights]).cpu().numpy()
    random_param_indices = sorted(np.random.choice(training_weights.shape[1], size=64, replace=False))

    generated, nearest_training = [], []
    for genid in range(3):
        l2_distances = np.linalg.norm(training_weights - generated_weights[genid], axis=1)
        top3_indices = np.argsort(l2_distances)[:3]
        generated.append(generated_weights[genid][random_param_indices])
        nearest_training.append([training_weights[idx][random_param_indices] for idx in top3_indices])
    
    draw(generated, nearest_training, os.path.abspath("figures"))