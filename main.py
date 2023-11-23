import csv
import random
from pathlib import Path
from typing import Dict, Callable, Union

import argh
import wandb
import numpy as np
import torch.optim
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from models import *
from dataset import SyntheticDataset
from evaluations import *


def train(model: nn.Module,
          device: str,
          train_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer) -> Dict[str, Union[int, float]]:

    batch_loss = 0.0
    train_loop = tqdm(train_loader, position=1, leave=False, desc='Train')
    with torch.set_grad_enabled(True):
        model.train()
        for batch_idx, (x, yt) in enumerate(train_loop):
            x = x.to(device)
            yt = yt.to(device)

            optimizer.zero_grad()
            yp = model(x)
            loss = criterion(yp, yt)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_loss += batch_loss

            train_loop.set_postfix(loss=batch_loss)

    return {'loss': batch_loss}


def evaluate(model: nn.Module,
             device: str,
             test_loader: DataLoader,
             evaluation: Dict[str, Callable]) -> Dict[str, Union[int, float]]:

    learning_data = {}
    for ev_name in evaluation.keys():
        learning_data[ev_name] = 0.0
    test_loop = tqdm(test_loader, position=1, leave=False, desc='Test')
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, yt) in enumerate(test_loop):
            x = x.to(device)
            yt = yt.to(device)

            yp = model(x)

            for ev_name, ev_fn in evaluation.items():
                learning_data[ev_name] += ev_fn(yp, yt)

    return learning_data


@argh.arg("epochs", type=int)
@argh.arg("model-name", type=str, choices=['unet', 'runet'])
@argh.arg("--use-cuda", default=True)
@argh.arg("--batch-size", type=int, default=64)
@argh.arg("--num-workers", type=int, default=4)
@argh.arg("--checkpoint-epoch", type=int, default=10)
@argh.arg("--save-path", type=Path, default=None)
@argh.arg("--seed", type=int, default=None)
def main(epochs: int,
         model_name: str,
         use_cuda: bool = True,
         batch_size=64,
         num_workers=4,
         checkpoint_epoch: int = 10,
         save_path: Path = None,
         seed: int = None):

    # Set environment variable WANDB_MODE=disabled
    # Remember to set WANDB_API_KEY as an environment variable
    wandb.login()

    device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"

    wandb.init(project="upsegmentation",
               config={
                   "epochs": epochs,
                   "model": model_name,
                   "device": device,
                   "batch_size": batch_size,
                   "num_workers": num_workers,
                   "seed": seed,
               })

    if wandb.run and save_path is None:
        save_path = wandb.run.dir

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    if model_name == 'unet':
        model = UNet(1, 2)
    elif model_name == 'runet':
        model = RUNet(1, 2)
    else:
        raise ValueError

    model = model.to(device)

    data = SyntheticDataset('data/generated/png', transforms.ToTensor())

    train_len = int(0.8 * len(data))
    test_len = len(data) - train_len

    train_data, test_data = torch.utils.data.random_split(data, [train_len, test_len])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    evaluations = {
        'true_positives': count_true_positives,
        'true_negatives': count_true_negatives,
        'false_positives': count_false_positives,
        'false_negatives': count_false_negatives,
    }

    learning_curve = []

    epoch_loop = tqdm(range(epochs), position=0, desc='Epoch')
    for epoch in epoch_loop:
        learning_data = {'epoch': epoch}

        learning_data.update({f"train_{k}": v
                              for k, v in train(model, device, train_loader, criterion, optimizer).items()})
        learning_data.update({f"eval_{k}": v
                              for k, v in evaluate(model, device, test_loader, evaluations).items()})

        learning_curve.append(learning_data)
        epoch_loop.set_postfix(learning_data)
        wandb.log(learning_data)

        if (epoch % checkpoint_epoch) == (checkpoint_epoch - 1):
            torch.save(model.state_dict(), save_path / f"{model_name}-checkpoint-{epoch:40d}.pth")

    torch.save(model.state_dict(), save_path / f"{model_name}.pth")

    with open(save_path / "learning_curve.tsv", "w", newline="") as f:
        fields = list(learning_curve[0].keys())
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(learning_curve)

    wandb.finish()


if __name__ == "__main__":
    argh.dispatch_command(main)
