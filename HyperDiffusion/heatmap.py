import numpy as np
import os
import sys
import torch
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_heatmap import draw

if __name__ == "__main__":
    training_weights = torch.load("./data/training_weights.pth").cuda()
    gen_weights = torch.load("./data/gen_weights.pth").cuda()
    random_param_indices = sorted(np.random.choice(training_weights.shape[1], size=64, replace=False))

    generated, nearest_training = [], []
    for generated_id in range(3):
        target = gen_weights[generated_id]
        target_expanded = target.expand(training_weights.size(0), -1)
        top3_distances_and_indices = [(float('inf'), -1)] * 3
        batch_size = 128

        for start in tqdm(range(0, training_weights.size(0), batch_size), mininterval=2):
            end = min(start + batch_size, training_weights.size(0))
            batch_training_weights = training_weights[start:end]
            l2_distances = torch.norm(batch_training_weights - target_expanded[start:end], dim=1)
            batch_top3_training_indices = l2_distances.argsort()[:3]

            for batch_top_training_index in batch_top3_training_indices:
                training_index = start + batch_top_training_index
                distance = l2_distances[batch_top_training_index].item()

                # update top 3 if current distance is in the top 3
                if distance < top3_distances_and_indices[-1][0]:
                    top3_distances_and_indices[-1] = (distance, training_index)
                    # sort the top 3 distances in ascending order
                    top3_distances_and_indices = sorted(top3_distances_and_indices, key=lambda x: x[0])

        generated.append(target.cpu().numpy()[random_param_indices])
        nearest_training.append([training_weights[v].cpu().numpy()[random_param_indices] for _, v in top3_distances_and_indices])
    
    draw(generated, nearest_training, os.path.abspath("figures"))