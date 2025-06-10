import matplotlib as mpl
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
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

label_size = 34

def generate_pca_grid(images, resolution=300):
    """
    Given a batch of (C*H*W) images,
    1) flattens & scales them,
    2) projects them to 2-D via PCA,
    3) builds a regular grid in that PCA space (with padding on each side),
    4) inverts the transform back to original image space,
    5) returns a (grid_size^2 * C * H * W) image tensor ready for model evaluation.
    """
    shape = images.shape[1:]
    images_flat = images.view(images.shape[0], -1).cpu().numpy()
    scaler = StandardScaler().fit(images_flat)
    images_scaled = scaler.transform(images_flat)

    pca = PCA(n_components=2)
    images_pca = pca.fit_transform(images_scaled)

    x_min, x_max = images_pca[:, 0].min() - 1, images_pca[:, 0].max() + 1
    y_min, y_max = images_pca[:, 1].min() - 1, images_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_original = pca.inverse_transform(grid)
    grid_original = scaler.inverse_transform(grid_original)
    return torch.tensor(grid_original, dtype=torch.float32).view(-1, *shape)

def draw(grid_preds, resolution, save_dir):
    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    all_contours = []
    axes[0, 0].set_title("generated", fontsize=label_size)
    axes[0, 1].set_title("training", fontsize=label_size)
    for generated_id in range(3):
        ax = axes[generated_id, 0]
        contour = ax.contourf(*np.meshgrid(range(resolution), range(resolution)), grid_preds[generated_id][0].reshape(resolution, resolution), alpha=0.3, levels=np.arange(11) - 0.5, cmap="tab10")
        ax.set_aspect("equal")
        all_contours.append(contour)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = axes[generated_id, 1]
        contour = ax.contourf(*np.meshgrid(range(resolution), range(resolution)), grid_preds[generated_id][1].reshape(resolution, resolution), alpha=0.3, levels=np.arange(11) - 0.5, cmap="tab10")
        ax.set_aspect("equal")
        all_contours.append(contour)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(right=0.75, wspace=0.05)
    fig.supxlabel("PCA dimension 1", fontsize=label_size, y=0.05, x=0.44)
    fig.supylabel("PCA dimension 2", fontsize=label_size, x=0.04)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "decision_boundary.png"), bbox_inches='tight')
    plt.close()