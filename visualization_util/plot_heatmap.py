import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
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

label_size = 48
tick_size = 36

def draw(generated, nearest_training, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(18, 8))
    
    for generated_id in range(3):
        ax = axes[generated_id]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        all_rows = [generated[generated_id]] + nearest_training[generated_id]
        all_concat = np.concatenate([arr.flatten() for arr in all_rows])
        global_min, global_max = all_concat.min(), all_concat.max()

        # the top row shows weights of a generated model
        top_ax = ax.inset_axes([0, 0.77, 1.0, 0.23])
        top_ax.imshow(all_rows[0:1], cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
        top_ax.set_xticks([]); top_ax.set_yticks([]); top_ax.set_frame_on(False)
        # add red border to the top row
        top_ax.add_patch(patches.Rectangle((0,0),1,1, transform=top_ax.transAxes,
                                        fill=False, edgecolor='red', linewidth=8))

        # the bottom block (three rows together) shows the weights of the 3 nearest training models
        bot_ax = ax.inset_axes([0, 0, 1.0, 0.23 * 3])
        bot_im = bot_ax.imshow(all_rows[1:], cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
        bot_ax.set_xticks([]); bot_ax.set_yticks([]); bot_ax.set_frame_on(False)
        
        for frac in (1/3, 2/3):
            bot_ax.hlines(y=frac, xmin=0, xmax=1, transform=bot_ax.transAxes,
                          color='white', linewidth=4, zorder=1)
        bot_ax.add_patch(patches.Rectangle((0,0),1,1, transform=bot_ax.transAxes,
                                           fill=False, edgecolor='black', linewidth=5, zorder=2))

        cbar = fig.colorbar(bot_im, ax=ax, orientation='vertical', location='left',
                            pad=0.01, aspect=10)
        cbar.ax.tick_params(labelsize=tick_size, labelleft=True, labelright=False,
                            length=8, width=2)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig.supylabel('parameter value', fontsize=label_size, x=0)
    fig.subplots_adjust(hspace=0.3, left=-0.05, right=0.995, top=0.98, bottom=0.02)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "heatmap.png"))
    plt.close()