import os
import json
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10 as Dataset
try:  # when name if main
    from model import Model
except ImportError:
    from .model import Model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_file, "r") as f:
        additional_config = json.load(f)
    config = {
        "dataset_root": "from_additional_config",
        "batch_size": 128,
        "num_workers": 4,
        "learning_rate": 0.0001,
        "weight_decay": 5e-4,
        "epochs": 1,  # Changed to 1 as we're only doing one epoch
        "save_learning_rate": 0.0001,
        "total_save_number": 300,
        "tag": os.path.basename(os.path.dirname(__file__)),
        "freeze_epochs": 0,
        "seed": 40
    }
    config.update(additional_config)
    return config


def get_data_loaders(config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = Dataset(root=config["dataset_root"], split="train", download=True, transform=train_transform)
    test_dataset = Dataset(root=config["dataset_root"], split="test", download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)
    return train_loader, test_loader


def get_optimizer_and_scheduler(model, config):
    trainable_params = model.parameters()
    optimizer = optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(get_data_loaders(config)[0]), eta_min=config["save_learning_rate"])
    return optimizer, scheduler


@torch.no_grad()
def test(model, test_loader, device):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicts = []
    pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        all_targets.extend(targets.cpu().tolist())
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.cpu().tolist())
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
        pbar.set_postfix({'Loss': f'{test_loss / (pbar.n + 1):.3f}', 'Acc': f'{100. * correct / total:.2f}%'})
    loss = test_loss / len(test_loader)
    acc = correct / total
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.4f}")
    return loss, acc, all_targets, all_predicts


def save_checkpoint(model, batch_idx, acc, config):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(model.state_dict(),
               f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{config['seed']:04d}_{config['tag']}.pth")
    print(f"Saved: checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{config['seed']:04d}_{config['tag']}.pth")


config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(config['seed'])

# load dataset
train_loader, test_loader = get_data_loaders(config)
# load model
model = Model().to(device)
state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained.pth"),
                        map_location=device, weights_only=True)
model.load_state_dict(state_dict)
# get optimizer
optimizer, scheduler = get_optimizer_and_scheduler(model, config)


if __name__ == "__main__":
    print("Initial test:")
    test(model, test_loader, device)
    total_batches = len(train_loader)
    save_interval = 1
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    ckpt_num = 0
    for j in range(1000):
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # Save checkpoint at regular intervals
            if ((batch_idx + 1) % save_interval == 0 or batch_idx == total_batches - 1) and batch_idx > 0:
                loss, acc, _, _ = test(model, test_loader, device)
                # loss, acc = 1., 1.
                save_checkpoint(model, ckpt_num, acc, config)
                ckpt_num += 1
            pbar.set_postfix({'Loss': f'{loss:.3f}'})
            if ckpt_num >= config["total_save_number"]:
                print("Fine-tuning completed.")
                exit(0)
