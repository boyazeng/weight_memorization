import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import torch
mpl.rcParams["xtick.direction"]="in"
mpl.rcParams["ytick.direction"]="in"
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
mpl.rcParams["xtick.major.width"]=1.5
mpl.rcParams["ytick.major.width"]=1.5
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["helvetica", "DejaVu Sans"],
})

label_size = 34
tick_size = 26
legend_size = 30

def calculate_min_distances(training_weights, target_weights, batch_size=1024):
    """
    Calculate the minimum L2 distance from each target checkpoint to all training checkpoints.
    If the training and target checkpoints are the same, the self-distance (always 0) is ignored.

    Args:
        training_weights (Tensor): [num_training, #dimensions] training checkpoints.
        target_weights (Tensor): [num_target, #dimensions] target checkpoints.

    Returns:
        Tensor[num_target]: minimum L2 distance for each target checkpoint.
    """
    training_weights = training_weights.to(torch.float32)
    target_weights = target_weights.to(torch.float32)

    min_distances = []

    for i in range(0, target_weights.size(0), batch_size):
        batch = target_weights[i:i+batch_size]
        # compute pairwise L2 distances
        dists = torch.cdist(batch, training_weights, p=2) # shape = [batch_size, num_training]

        # mask self-distances if training and target are the same
        if training_weights is target_weights:
            indices = torch.arange(i, i + batch.size(0), device=batch.device)
            dists[torch.arange(batch.size(0)), indices] = float('inf')

        min_distances.append(torch.min(dists, dim=1).values) # shape = [batch_size]

    return torch.cat(min_distances)

def draw(min_distances_training, min_distances_generated, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if isinstance(min_distances_training, torch.Tensor):
        min_distances_training = min_distances_training.cpu().numpy()
    if isinstance(min_distances_generated, torch.Tensor):
        min_distances_generated = min_distances_generated.cpu().numpy()

    min_val = min(np.min(min_distances_training), np.min(min_distances_generated))
    max_val = max(np.max(min_distances_training), np.max(min_distances_generated))
    bin_width = (max_val - min_val) / 15
    bin_edges = np.arange(min_val, max_val + bin_width, bin_width)

    bin_count_training, _ = np.histogram(min_distances_training, bins=bin_edges, density=False)
    bin_count_generated, _ = np.histogram(min_distances_generated, bins=bin_edges, density=False)

    bin_count_training = bin_count_training / bin_count_training.sum() * 100
    bin_count_generated = bin_count_generated / bin_count_generated.sum() * 100
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.bar(bin_centers, bin_count_generated, width=bin_width, label='generated', color="red", alpha=0.8)
    ax.bar(bin_centers, bin_count_training, width=bin_width, label='training', color="tab:blue", alpha=0.8)

    plt.xlabel(r"min $L_2$ to training weights", fontsize=label_size)
    plt.ylabel("% of models", fontsize=label_size)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[['top', 'right']].set_visible(False)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "min_distance_to_training.png"), bbox_inches='tight')
    plt.close()