import os; os.chdir("../method")
import sys; sys.path.append("../method")
import argparse
import numpy as np
import torch
import importlib
item = importlib.import_module(f"dataset.main.cifar100_resnet18.finetune")
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_dataset = CIFAR100(root="~/.cache/p-diff/datasets", train=False, download=True, transform=test_transform)
loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)
model = item.model
test = item.test

config = {
    "training_path": "./dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": "./dataset/main/cifar100_resnet18/generated",
    "noise_path": "./dataset/main/cifar100_resnet18/noise",
    "average_path": "./dataset/main/cifar100_resnet18/average",
    "gaussian_path": "./dataset/main/cifar100_resnet18/gaussian",
}

@torch.no_grad()
def compute_wrong_indices(diction):
    model.load_state_dict(diction, strict=False)
    model.eval()
    _, acc, all_targets, all_predicts = test(model=model, test_loader=loader, device="cuda")
    incorrect_indices = torch.logical_not(torch.eq(torch.tensor(all_targets), torch.tensor(all_predicts)))
    return incorrect_indices, acc, all_predicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["training", "generated", "noise", "average", "gaussian"])
    parser.add_argument("--noise_amplitude", type=str, choices=['0.06', '0.12'])
    args = parser.parse_args()

    checkpoint_path = config[f"{args.type}_path"]
    if args.type == "noise":
        checkpoint_path = os.path.join(checkpoint_path, str(args.noise_amplitude))

    full_accs, full_incorrect_indices = [], []
    ckpts = sorted([os.path.join(checkpoint_path, i) for i in os.listdir(checkpoint_path)])
    for ckpt in ckpts:
        diction = torch.load(ckpt, map_location="cuda", weights_only=True)
        incorrect_indices, acc, all_predicts = compute_wrong_indices(diction)
        full_accs.append(acc)
        full_incorrect_indices.append(incorrect_indices)

    suffix = args.noise_amplitude if args.type == "noise" else ""
    os.makedirs("../data", exist_ok=True)
    np.save(f"../data/{args.type}{suffix}_accs.npy", full_accs)
    np.save(f"../data/{args.type}{suffix}_incorrect_indices.npy", full_incorrect_indices)