import os
import sys
import torch

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_min_distance_to_training import calculate_min_distances, draw

config = {
    "training_path": f"./method/dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": f"./method/dataset/main/cifar100_resnet18/generated",
}

def dict_to_flat(diction):
    return torch.cat([diction[key].flatten() for key in sorted(diction.keys())]).cpu().numpy()

if __name__ == "__main__":
    training_ckpts = [os.path.join(config["training_path"], i) for i in os.listdir(config["training_path"])]
    training_ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in training_ckpts]
    training_weights = torch.tensor([dict_to_flat(i) for i in training_ckpts])

    generated_ckpts = [os.path.join(config["generated_path"], i) for i in os.listdir(config["generated_path"])]
    generated_ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in generated_ckpts]
    gen_weights = torch.tensor([dict_to_flat(i) for i in generated_ckpts])

    min_distances_training = calculate_min_distances(training_weights, training_weights)
    min_distances_generated = calculate_min_distances(training_weights, gen_weights)
    draw(min_distances_training, min_distances_generated, os.path.abspath("figures"))