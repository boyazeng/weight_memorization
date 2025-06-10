import os
import sys
import torch
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_scatter_plot import draw, max_similarity_to_training, randomly_select_100

if __name__ == "__main__":
    training_incorrect_indices = torch.load("./data/training_incorrect_indices.pt").cpu().numpy()
    novelty = []
    performance = []
    types = ["training", "generated", "noise0.05", "noise0.1"]
    for type in types:
        incorrect_indices = torch.load(f"./data/{type}_incorrect_indices.pt").cpu().numpy()
        accs = torch.load(f"./data/{type}_accs.pt").cpu().numpy()
        similarity = max_similarity_to_training(training_incorrect_indices, incorrect_indices, batch_size=8)
        similarity, accs = randomly_select_100(similarity, accs)
        novelty.append(similarity)
        performance.append(accs)
    draw(novelty, performance, "maximum similarity", "accuracy (%)", types, os.path.abspath("figures/performance_vs_novelty.png"))