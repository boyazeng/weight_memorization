import json
from method.src.ghrp.model_definitions.def_net import NNmodule
from method.src.ghrp.checkpoints_to_datasets.dataset_auxiliaries import vector_to_checkpoint
import numpy as np
import os
from pathlib import Path
import random
import sys
import torch
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_decision_boundary import draw, generate_pca_grid

random.seed(2)
class_indices = random.sample(range(10), 5)

@torch.no_grad()
def main():
    data_dir = Path("./method/data")
    gen_weights = torch.tensor(np.load("./data/generated_weights.npy"), device="cuda")
    reconstructed_weights = torch.tensor(np.load("./data/reconstructed_weights.npy"), device="cuda")

    model_config = json.load((data_dir / "hyper_representations/svhn/config_zoo.json").open("r"))
    base_model = NNmodule(model_config, cuda=True)
    checkpoint_base = {k: v.cpu().clone() for k, v in base_model.model.state_dict().items()}

    dataset = torch.load(data_dir / "zoos/svhn/image_dataset.pt", weights_only=False)
    images = dataset["testset"].tensors[0]
    grid_images_tensor = generate_pca_grid(images)

    dists = torch.cdist(reconstructed_weights, gen_weights[:3], p=2)
    nearest_training_indices = torch.argmin(dists, dim=0)

    grid_preds = [[] for _ in range(3)]
    for generated_id in range(3):
        generated_ckpt = vector_to_checkpoint(
            checkpoint=checkpoint_base,
            vector=gen_weights[generated_id],
            layer_lst=[[0,"conv2d"],[3,"conv2d"],[6,"conv2d"],[9,"fc"],[11,"fc"]],
            use_bias=True,
        )
        base_model.model.load_state_dict(generated_ckpt)
        grid_preds[generated_id].append(base_model.model(grid_images_tensor.cuda()).argmax(dim=1).cpu().numpy())

        nearest_reconstructed_ckpt = vector_to_checkpoint(
            checkpoint=checkpoint_base,
            vector=reconstructed_weights[nearest_training_indices[generated_id]],
            layer_lst=[[0,"conv2d"],[3,"conv2d"],[6,"conv2d"],[9,"fc"],[11,"fc"]],
            use_bias=True,
        )
        base_model.model.load_state_dict(nearest_reconstructed_ckpt)
        grid_preds[generated_id].append(base_model.model(grid_images_tensor.cuda()).argmax(dim=1).cpu().numpy())

    draw(grid_preds, 300, "figures")

if __name__ == "__main__":
    main()