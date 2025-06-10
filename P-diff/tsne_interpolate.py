import numpy as np
import os
import random
from sklearn.manifold import TSNE
import sys
import torch

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_scatter_plot import draw

config = {
    "training_path": "./method/dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": "./method/dataset/main/cifar100_resnet18/generated",
    "averaged_path": "./method/dataset/main/cifar100_resnet18/average",
    "gaussian_path": "./method/dataset/main/cifar100_resnet18/gaussian",
}

def dict_to_flat(diction):
    return torch.cat([item.flatten() for item in diction.values()]).cpu().numpy()

if __name__ == "__main__":
    names = ["training", "generated", "averaged", "gaussian"]
    model_weights = [[] for _ in range(len(names))]
    for i, name in enumerate(names):
        ckpts = [os.path.join(config[name + "_path"], i) for i in os.listdir(config[name + "_path"])]
        ckpts = random.sample(ckpts, 100)
        ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in ckpts]
        model_weights[i] = np.array([dict_to_flat(i) for i in ckpts])

    data = np.vstack(model_weights)
    labels = np.array([i for i, pt in enumerate(model_weights) for _ in range(pt.shape[0])])
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)
    tsne_dim_1, tsne_dim_2 = [], []
    for i in range(len(model_weights)):
        subset = data_2d[labels == i]
        tsne_dim_1.append(subset[:, 0])
        tsne_dim_2.append(subset[:, 1])

    draw(tsne_dim_1, tsne_dim_2, "t-SNE dim 1", "t-SNE dim 2", names, os.path.abspath("figures/tsne_interpolate.png"))