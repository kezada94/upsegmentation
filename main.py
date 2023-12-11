import csv
import random
from pathlib import Path
from typing import Dict, Callable, Any

import argh
import wandb
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from models import *
from dataset import SyntheticDataset
from evaluations import *
from plot import plot_roc_and_samples


def torch_2_array(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    x = np.repeat(x[:, 0, :, :][:, None, :, :], 3, axis=1)
    x = np.transpose(x, (0, 2, 3, 1))
    x = (np.clip(x, 0, 1) * 255).astype('uint8')
    return x


def train(model: nn.Module,
          device: str,
          loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          evaluation: Dict[str, Callable]) -> Dict[str, Any]:

    learning_data = {ev_name: 0.0 for ev_name in evaluation.keys()}
    learning_data['loss'] = 0.0
    loop = tqdm(loader, position=1, leave=True, desc='Train')
    with torch.set_grad_enabled(True):
        model.train()
        for batch_idx, (x, yt) in enumerate(loop):
            x = x.to(device)
            yt = yt.to(device)

            optimizer.zero_grad()
            yp = model(x)
            loss = criterion(yp, yt)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()

            loop.set_postfix(loss=batch_loss)
            learning_data['loss'] += batch_loss

            for ev_name, ev_fn in evaluation.items():
                learning_data[ev_name] += ev_fn(yp, yt)

    return learning_data


def test(model: nn.Module,
         device: str,
         loader: DataLoader,
         evaluation: Dict[str, Callable]) -> Dict[str, Any]:

    learning_data = {ev_name: 0.0 for ev_name in evaluation.keys()}
    loop = tqdm(loader, position=1, leave=True, desc='Test')
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, yt) in enumerate(loop):
            x = x.to(device)
            yt = yt.to(device)

            yp = model(x)

            for ev_name, ev_fn in evaluation.items():
                learning_data[ev_name] += ev_fn(yp, yt)

    return learning_data


def plot_checkpoint(model: nn.Module,
                    device: str,
                    loader: DataLoader,
                    image_to_show=-1) -> Dict[str, Any]:
    _fpr = []
    _tpr = []
    _auc = []
    display_x = None
    display_yt = None
    display_yp = None

    loop = tqdm(loader, position=1, leave=True, desc='Test')
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, yt) in enumerate(loop):
            x = x.to(device)
            yt = yt.to(device)

            yp = F.softmax(model(x), dim=1)

            x = torch_2_array(x)
            yt = torch_2_array(yt)
            yp = torch_2_array(yp)

            for i in range(x.shape[0]):
                fpr, tpr, _ = roc_curve(yt[i].ravel(), yp[i].ravel(), pos_label=1)
                _fpr.append(fpr)
                _tpr.append(tpr)
                _auc.append(auc(fpr, tpr))

                if image_to_show == batch_idx + i:
                    display_x = x[i]
                    display_yt = yt[i]
                    display_yp = yp[i]

    mean_fpr = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(_fpr, _tpr)], axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)

    fig, ax = plot_roc_and_samples(
        display_x,
        display_yt,
        display_yp,
        _fpr,
        _tpr,
        mean_fpr,
        mean_tpr,
        mean_auc,
        image_to_show)

    return {'plot': fig, 'mean_auc': mean_auc, 'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr}


@argh.arg("epochs", type=int)
@argh.arg("model-name", type=str, choices=['unet', 'runet', 'runetfc'])
@argh.arg("--use-cuda", default=True)
@argh.arg("--batch-size", type=int, default=64)
@argh.arg("--learning-rate", type=float, default=1e-4)
@argh.arg("--num-workers", type=int, default=4)
@argh.arg("--checkpoint-epoch", type=int, default=10)
@argh.arg("--save-path", type=Path, default=None)
@argh.arg("--seed", type=int, default=None)
def main(epochs: int,
         model_name: str,
         use_cuda: bool = True,
         batch_size=64,
         learning_rate=1e-4,
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
                   "learning_rate": learning_rate,
                   "num_workers": num_workers,
                   "seed": seed,
               })

    # Check if save_path is None and wandb is not disabled
    # If wandb is not disabled, save_path is set to wandb.run.dir
    if save_path is None and wandb.run is not None:
        save_path = Path(wandb.run.dir)

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    print('Saving checkpoints to', save_path)

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
    elif model_name == 'runetfc':
        model = RUNetFC(1, 2)
    else:
        raise ValueError

    model = model.to(device)

    train_data = SyntheticDataset('data/generated/png/train', transforms.ToTensor())
    test_data = SyntheticDataset('data/generated/png/test', transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    evaluations = {
        'accuracy': accuracy,
    }

    learning_curve = []

    epoch_loop = tqdm(range(epochs), position=0, desc='Epoch', leave=True)
    for epoch in epoch_loop:
        learning_data = {'epoch': epoch}

        learning_data.update({f"train_{k}": v
                              for k, v in train(model, device, train_loader, criterion, optimizer, evaluations).items()
                              })
        learning_data.update({f"test_{k}": v
                              for k, v in test(model, device, test_loader, evaluations).items()
                              })

        learning_curve.append(learning_data)
        epoch_loop.set_postfix(learning_data)

        if save_path and (epoch % checkpoint_epoch) == (checkpoint_epoch - 1):
            checkpoint_name = f"{model_name}-checkpoint-{epoch:04d}.pth"
            torch.save(model.state_dict(), save_path / checkpoint_name)
            learning_data.update(plot_checkpoint(model, device, test_loader, image_to_show=0))

        wandb.log(learning_data)

    if save_path:
        torch.save(model.state_dict(), save_path / f"{model_name}.pth")

        # with open(save_path / "learning_curve.tsv", "w", newline="") as f:
        #     fields = list(learning_curve[0].keys())
        #     writer = csv.DictWriter(f, fields, delimiter="\t")
        #     writer.writeheader()
        #     writer.writerows(learning_curve)

    wandb.finish()


if __name__ == "__main__":
    argh.dispatch_command(main)
