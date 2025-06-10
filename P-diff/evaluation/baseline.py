import os; os.chdir("../method")
import sys; sys.path.append("../method")
import argparse
import random
from scipy.stats import norm
import torch

config = {
    "training_path": "./dataset/main/cifar100_resnet18/checkpoint",
    "noise_path": "./dataset/main/cifar100_resnet18/noise",
    "average_path": "./dataset/main/cifar100_resnet18/average",
    "gaussian_path": "./dataset/main/cifar100_resnet18/gaussian",
}

structure = {
    "model.layer4.1.bn1.weight": [0, 512],
    "model.layer4.1.bn1.bias": [512, 512*2],
    "model.layer4.1.bn2.weight": [512*2, 512*3],
    "model.layer4.1.bn2.bias": [512*3, 512*4],
}

def flatten(diction):
    vector = []
    for key in structure:
        value = diction[key]
        vector.append(value.flatten())
    return torch.cat(vector, dim=0)

def add_noise(diction, noise_amplitude):
    for key in diction:
        assert ("running_var" not in key) and ("num_batches_tracked" not in key)
        diction[key] = diction[key] + torch.randn_like(diction[key]) * noise_amplitude
    return diction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["noise", "average", "gaussian"])
    parser.add_argument("--noise_amplitude", type=float, choices=[0.06, 0.12])
    args = parser.parse_args()

    training_ckpts = sorted([os.path.join(config["training_path"], i) for i in os.listdir(config["training_path"])])
    training_ckpts = [torch.load(i, map_location="cuda", weights_only=True) for i in training_ckpts]
    if args.type == "noise":
        config["noise_path"] = os.path.join(config["noise_path"], str(args.noise_amplitude))
        os.makedirs(config["noise_path"], exist_ok=True)
        noised_accs, noised_flags = [], []
        for i, ckpt in enumerate(training_ckpts):
            ckpt = add_noise(ckpt, args.noise_amplitude)
            torch.save(ckpt, os.path.join(config["noise_path"], f"{i}.pth"))
    
    elif args.type == "average":
        os.makedirs(config["average_path"], exist_ok=True)
        for i in range(100):
            sampled_ckpts = random.sample(training_ckpts, 16)
            averaged_diction = None
            for ckpt in sampled_ckpts:
                diction = ckpt
                if averaged_diction is None:
                    averaged_diction = diction
                else:
                    for key in diction:
                        averaged_diction[key] += diction[key]
            for key in averaged_diction:
                averaged_diction[key] /= 16
            torch.save(averaged_diction, os.path.join(config["average_path"], f"{i}.pth"))

    elif args.type == "gaussian":
        os.makedirs(config["gaussian_path"], exist_ok=True)
        training_ckpts = torch.stack([flatten(i) for i in training_ckpts])
        means = training_ckpts.mean(dim=0).cpu().numpy()
        stds = training_ckpts.std(dim=0).cpu().numpy()
        for i in range(100):
            sampled_vector = torch.tensor(norm.rvs(loc=means, scale=stds))
            sampled_diction = {}
            for key, (start, end) in structure.items():
                sampled_diction[key] = sampled_vector[start:end]
            torch.save(sampled_diction, os.path.join(config["gaussian_path"], f"{i}.pth"))