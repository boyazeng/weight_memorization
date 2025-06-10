import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import os
import torch
mpl.rcParams["xtick.direction"]="in"
mpl.rcParams["ytick.direction"]="in"
mpl.rcParams["xtick.major.size"]=5
mpl.rcParams["ytick.major.size"]=5
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["helvetica", "DejaVu Sans"],
})

label_size = 24
tick_size = 18
marker_size = 50
legend_size = 20

def max_similarity_to_training(training_incorrect_indices, target_incorrect_indices, batch_size=128):
    """
    Compute maximum similarity (IoU of incorrect predictions) between each target and all training checkpoints.

    Args:
        training_incorrect_indices (Tensor): [num_training, testset_size] incorrect prediction mask for training checkpoints.
        target_incorrect_indices (Tensor): [num_target, testset_size] incorrect prediction mask for target checkpoints.

    Returns:
        Tensor[num_target]: maximum similarity for each target checkpoint.
    """
    training_incorrect_indices = torch.tensor(training_incorrect_indices, dtype=torch.bool, device='cuda')
    target_incorrect_indices = torch.tensor(target_incorrect_indices, dtype=torch.bool, device='cuda')
    num_training = training_incorrect_indices.size(0)

    ratio_matrix = []
    for start_idx in range(0, num_training, batch_size):
        end_idx = min(start_idx + batch_size, num_training)
        training_batch = training_incorrect_indices[start_idx:end_idx]
        training_expanded = training_batch.unsqueeze(1) # shape = [batch_size, 1, testset_size]
        target_expanded = target_incorrect_indices.unsqueeze(0) # shape = [1, num_target, testset_size]
        overlapped_errors = torch.sum(training_expanded & target_expanded, dim=2) # shape = [batch_size, num_target]
        unique_errors = torch.sum(training_expanded | target_expanded, dim=2) # shape = [batch_size, num_target]
        ratios = overlapped_errors.float() / (unique_errors.float() + 1e-8) # shape = [batch_size, num_target]
        ratio_matrix.append(ratios)

    full_ratio_matrix = torch.cat(ratio_matrix, dim=0)
    if torch.equal(training_incorrect_indices, target_incorrect_indices):
        full_ratio_matrix.fill_diagonal_(0)
    max_ratios = torch.max(full_ratio_matrix, dim=0)[0]

    return max_ratios.cpu().numpy()

def randomly_select_100(a, b):
    assert len(a) == len(b)
    indices = np.random.choice(len(a), 100, replace=False)
    return a[indices], b[indices]

def draw(x_values, y_values, x_label, y_label, types, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    colors = ["tab:blue", "red", "#493d8c", "orange"]
    
    for x_val, y_val, color, type in zip(x_values, y_values, colors, types):
        ax.scatter(x_val, y_val, s=marker_size, label=type, color=color, alpha=0.8, edgecolor='black', linewidth=0.1)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.spines[['top', 'right']].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}' if x == 0 else f'{x:.2f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}' if y == 0 else f'{y:.2f}'))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.9),
        ncol=4,
        columnspacing=0.1,
        handletextpad=-0.4,
        fontsize=legend_size,
        frameon=True,
        markerscale=1.2,
        labelspacing=0.2,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.73)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()