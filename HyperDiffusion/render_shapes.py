import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import trimesh
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

def draw_mesh(load_path):
    mesh = trimesh.load(load_path)
    bounding_box = mesh.bounding_box.bounds
    min_bound = bounding_box[0]
    max_bound = bounding_box[1]
    center = (min_bound + max_bound) / 2
    scale = (max_bound - min_bound).max() * 0.52
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        color='#d3d3d3',
        edgecolor='none',
        alpha=1.0
    )
    ax.axis('off')
    ax.set_xlim(center[0] - scale / 2, center[0] + scale / 2)
    ax.set_ylim(center[1] - scale / 2, center[1] + scale / 2)
    ax.set_zlim(center[2] - scale / 2, center[2] + scale / 2)
    ax.view_init(elev=-10, azim=40)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    plt.close(fig)
    return np.rot90(img, k=1)

if __name__ == "__main__":
    gen_weights = torch.load("./data/gen_weights.pth").cuda()
    training_weights = torch.load("./data/training_weights.pth").cuda()

    gen_mesh_dir = f"./data/gen_meshes"
    train_mesh_dir = f"./data/train_meshes"
    dists = torch.cdist(gen_weights[:3], training_weights, p=2)
    nearest_training_indices = torch.argmin(dists, dim=1)
    paths_gen = [f"{gen_mesh_dir}/{genid}.obj" for genid in range(3)]
    paths_train = [f"{train_mesh_dir}/mesh_{nearest_training_indices[genid]}.obj" for genid in range(3)]

    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    axes[0, 0].set_title("generated", fontsize=label_size)
    axes[0, 1].set_title("training", fontsize=label_size)
    for genid in range(3):
        ax = axes[genid, 0]
        ax.imshow(draw_mesh(paths_gen[genid]), aspect='equal')
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = axes[genid, 1]
        ax.imshow(draw_mesh(paths_train[genid]), aspect='equal')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.subplots_adjust(right=0.75, wspace=0.05)
    fig.supxlabel("PCA dimension 1", fontsize=label_size, y=0.05, x=0.44, color="none")
    fig.supylabel("PCA dimension 2", fontsize=label_size, x=0.04, color="none")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/render_shapes.png", bbox_inches='tight')
    plt.close()