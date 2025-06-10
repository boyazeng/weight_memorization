import os; os.chdir("../method")
import sys; sys.path.append("../method")
import argparse
import numpy as np
from pathlib import Path
from src.ghrp.model_definitions.def_FastTensorDataLoader import FastTensorDataLoader
from src.ghrp.model_definitions.def_net import NNmodule
from src.ghrp.checkpoints_to_datasets.dataset_auxiliaries import vector_to_checkpoint
import torch
import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["reconstructed", "generated", "noise"])
    parser.add_argument("--noise_amplitude", type=str, choices=['0.02', '0.04'])
    args = parser.parse_args()
    if args.type == "noise":
        args.type += args.noise_amplitude

    data_dir = Path("./data")
    config = json.load(open(data_dir / "hyper_representations/svhn/config_zoo.json"))
    testset = torch.load(data_dir / "zoos/svhn/image_dataset.pt", weights_only=False)["testset"]
    testloader = FastTensorDataLoader(
        dataset=testset, batch_size=len(testset), shuffle=False
    )
    model_config = json.load(open(data_dir / "hyper_representations/svhn/config_zoo.json"))
    base_model = NNmodule(model_config, cuda=True)
    checkpoint_base = {k: v.cpu().clone() for k, v in base_model.model.state_dict().items()}

    weights = torch.tensor(np.load(f"../data/{args.type}_weights.npy"), device="cuda")
    full_accs = []
    full_incorrect_indices = []
    for idx in tqdm.tqdm(range(len(weights))):
        ckpt = vector_to_checkpoint(
            checkpoint=checkpoint_base,
            vector=weights[idx],
            layer_lst=[[0,"conv2d"],[3,"conv2d"],[6,"conv2d"],[9,"fc"],[11,"fc"]],
            use_bias=True,
        )
        base_model.model.load_state_dict(ckpt)
        test_acc, incorrect_indices = base_model.accs_and_incorrect_indices(testloader)
        full_accs.append(test_acc)
        full_incorrect_indices.append(incorrect_indices)
    np.save(f"../data/{args.type}_accs.npy", full_accs)
    np.save(f"../data/{args.type}_incorrect_indices.npy", full_incorrect_indices)
