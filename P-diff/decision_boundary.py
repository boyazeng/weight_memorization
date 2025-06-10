import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import numpy as np
import os
import random
import sys
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_decision_boundary import draw, generate_pca_grid

item = importlib.import_module(f"method.dataset.main.cifar100_resnet18.finetune")
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_dataset = CIFAR100(root="~/.cache/p-diff/datasets", train=False, download=True, 
                        transform=transforms.ToTensor())
random.seed(42)
class_indices = set(random.sample(range(100), 10))
images = torch.stack(
    [img for img, lbl in test_dataset if lbl in class_indices]
).cuda()

model = item.model

@torch.no_grad()
def get_predictions_within_classes(model, test_loader, device, class_indices):
    model.eval()
    all_targets = []
    all_predicts = []
    mask = torch.tensor([t in class_indices for t in range(100)], device=device)
    pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
        outputs = outputs[:, mask]
        all_targets.extend(targets.cpu().tolist())
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.cpu().tolist())
    return all_predicts

config = {
    "training_path": f"./method/dataset/main/cifar100_resnet18/checkpoint",
    "generated_path": f"./method/dataset/main/cifar100_resnet18/generated",
}

def dict_to_flat(diction):
    return torch.cat([item.flatten() for item in diction.values()])

@torch.no_grad()
def compute_predictions(diction, loader):
    model.load_state_dict(diction, strict=False)
    model.eval()
    predicts = get_predictions_within_classes(model=model, test_loader=loader, device="cuda", class_indices=class_indices)
    return predicts

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=300)
    args = parser.parse_args()

    training_dicts = [torch.load(os.path.join(config["training_path"], path)) for path in sorted(os.listdir(config["training_path"]))]
    generated_dicts = [torch.load(os.path.join(config["generated_path"], path)) for path in sorted(os.listdir(config["generated_path"]))]
    training_weights = torch.stack([dict_to_flat(item) for item in training_dicts])
    generated_weights = torch.stack([dict_to_flat(item) for item in generated_dicts])

    grid_images_tensor = generate_pca_grid(images)
    grid_images_list = [grid_images_tensor[i] for i in range(grid_images_tensor.shape[0])]

    with ThreadPoolExecutor(max_workers=16) as executor:
        transformed_images = list(executor.map(test_transform, grid_images_list))    
    grid_images_tensor = torch.stack(transformed_images)
    grid_dset = TensorDataset(grid_images_tensor, torch.zeros(grid_images_tensor.shape[0]))
    loader = DataLoader(grid_dset, batch_size=2048, shuffle=False, num_workers=16, pin_memory=True)

    grid_preds = [[] for _ in range(3)]
    for generated_id in range(3):
        l2_distances = (training_weights - generated_weights[generated_id]).norm(dim=1)
        nearest_training_index = np.argsort(l2_distances)[0]
        grid_preds[generated_id].append(np.array(compute_predictions(generated_dicts[generated_id], loader)))
        grid_preds[generated_id].append(np.array(compute_predictions(training_dicts[nearest_training_index], loader)))

    draw(grid_preds, args.resolution, "figures")

if __name__ == "__main__":
    main()