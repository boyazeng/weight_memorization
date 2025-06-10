import os
import torch
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
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

label_size = 42
tick_size = 34
legend_size = 40

config = {
    "training_path": f"./method/dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": f"./method/dataset/main/cifar100_resnet18/generated",
}

if __name__ == "__main__":
    training_ckpts = [os.path.join(config["training_path"], i) for i in os.listdir(config["training_path"])]
    training_ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in training_ckpts]
    training_ckpts = np.array([i['model.layer4.1.bn1.weight'].flatten().cpu().numpy() for i in training_ckpts])

    generated_ckpts = [os.path.join(config["generated_path"], i) for i in os.listdir(config["generated_path"])]
    generated_ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in generated_ckpts]
    generated_ckpts = np.array([i['model.layer4.1.bn1.weight'].flatten().cpu().numpy() for i in generated_ckpts])

    sampled_indices = sorted(np.random.choice(training_ckpts.shape[1], 5, replace=False))
    fig, axes = plt.subplots(1, 5, figsize=(30, 4.5), sharey=True)

    for ax, sample_index in zip(axes, sampled_indices):
        sns.kdeplot(training_ckpts[:, sample_index], label='training', color="tab:blue", linewidth=4, ax=ax)
        sns.kdeplot(generated_ckpts[:, sample_index], label='generated', color="red", linewidth=4, ax=ax)
        ax.set_xlabel("", fontsize=label_size)
        ax.set_ylabel("", fontsize=label_size)
        ax.spines[['top', 'right']].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_size)
        ax.xaxis.set_tick_params(pad=10)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_size)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(True, linestyle='--', color='gray', linewidth=1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        if ax != axes[0]:
            ax.tick_params(axis='y', length=0)

    fig.supxlabel("parameter value", fontsize=label_size, y=0.05)
    fig.supylabel("density", fontsize=label_size, y=0.6, x=0.015)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="upper right",
        fontsize=legend_size,
        frameon=True,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1,
        handlelength=2.0,
        handleheight=0.5,
    )
    plt.tight_layout(rect=[0.055, 0, 1, 1])
    plt.subplots_adjust(left=0.055)
    plt.savefig("figures/param_value_distribution.png", bbox_inches='tight')