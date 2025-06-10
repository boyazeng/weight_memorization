import os
import sys
import torch

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_min_distance_to_training import calculate_min_distances, draw

if __name__ == "__main__":
    training_weights = torch.load("./data/training_weights.pth").cuda()
    gen_weights = torch.load("./data/gen_weights.pth").cuda()
    min_distances_training = calculate_min_distances(training_weights, training_weights)
    min_distances_generated = calculate_min_distances(training_weights, gen_weights)
    draw(min_distances_training, min_distances_generated, os.path.abspath("figures"))