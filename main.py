import random
from pathlib import Path
from typing import Tuple, Dict, Any

import argh
import wandb
import numpy as np
import torch.optim
import torch.nn as nn
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import ImageOps
from skimage.transform import resize
from torch.utils.data import DataLoader

from models import *
from evaluations import *
from dataset import SyntheticDataset


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    path: Path,
                    name: str,
                    verbose: bool = True) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path / f'{name}.pth')

    if verbose:
        print(f'Checkpoint saved as {name}.pth')


def load_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    path: Path,
                    name: str,
                    verbose: bool = True) -> int:
    if verbose:
        print(f'Loading checkpoint from {name}.pth')
    checkpoint = torch.load(path / f'{name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def train(model: nn.Module,
          loader: DataLoader,
          evaluation: Evaluations,
          device: torch.device,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer) -> Dict[str, Any]:

    evaluation.reset()
    log_data = {'loss': 0.0}
    loop = tqdm(loader, desc='Train')
    with torch.set_grad_enabled(True):
        model.train()
        for batch_idx, (x, yt) in enumerate(loop):
            one_hot_yt = torch.nn.functional.one_hot(yt, -1).transpose(1, 4).squeeze(-1).to(torch.float32)

            x = x.to(device)
            yt = yt.to(device)
            one_hot_yt = one_hot_yt.to(device)

            optimizer.zero_grad()
            one_hot_yp = model(x)

            loss = criterion(one_hot_yp, one_hot_yt)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            log_data['loss'] += batch_loss

            yp = torch.argmax(one_hot_yp, dim=1)
            log_data.update(evaluation(yt, yp))
            loop.set_postfix(log_data)
    return log_data


def test(model: nn.Module,
         loader: DataLoader,
         evaluation: Evaluations,
         device: torch.device) -> Dict[str, Any]:
    class_labels = {0: "background", 1: "border"}
    evaluation.reset()
    log_data = {}
    loop = tqdm(loader, desc='Test')
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, yt) in enumerate(loop):
            x = x.to(device)
            yt = yt.to(device)
            yp = torch.argmax(model(x), dim=1)
            log_data.update(evaluation(yt, yp))

            if batch_idx == 0:
                log_data['prediction'] = []
                for img_idx in range(len(x)):
                    x_img = x[img_idx].detach().cpu().numpy()
                    yp_img = yp[img_idx].detach().cpu().numpy().astype(np.uint8()).squeeze()
                    yt_img = yt[img_idx].detach().cpu().numpy().astype(np.uint8()).squeeze()

                    x_img = np.transpose(x_img, (1, 2, 0))
                    x_img = resize(x_img, yp_img.shape, anti_aliasing=True)

                    log_data['prediction'].append(
                        wandb.Image(
                            x_img,
                            masks={
                                "predictions": {"mask_data": yp_img, "class_labels": class_labels},
                                "ground_truth": {"mask_data": yt_img, "class_labels": class_labels},
                            },
                            caption=f"Image {batch_idx + img_idx}",
                        )
                    )

            loop.set_postfix(log_data)
    return log_data


class CustomTransform:
    def __init__(self, device, n_classes, mean=0, std=1):
        self.device = device
        self.n_classes = n_classes
        self.mean = mean
        self.std = std

    def __call__(self, x, y):
        x = ImageOps.grayscale(x)
        x = TF.to_tensor(x)
        x = TF.normalize(x, mean=self.mean, std=self.std)
        x = x.to(torch.float32)

        y = TF.pil_to_tensor(y) / 255
        y = y.to(torch.uint8)
        y = y.to(torch.long)

        return x, y


@argh.arg("epochs", type=int)
@argh.arg("model-name", type=str, choices=['runet'])
@argh.arg("--optimizer", type=str, default='adam')
@argh.arg("--batch-size", type=int, default=64)
@argh.arg("--learning-rate", type=float, default=1e-4)
@argh.arg("--balance-classes", type=float, nargs=2, default=None)
@argh.arg("--use-cuda", default=True)
@argh.arg("--num-workers", type=int, default=4)
@argh.arg("--checkpoint-epoch", type=int, default=None)
@argh.arg("--save-path", type=Path, default=None)
@argh.arg("--seed", type=int, default=None)
def main(epochs: int,
         model_name: str,
         optimizer: str = 'adam',
         batch_size=64,
         learning_rate=1e-4,
         balance_classes: Tuple[float, float] = None,
         use_cuda: bool = True,
         num_workers=4,
         checkpoint_epoch: int = None,
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
                   "optimizer": optimizer,
                   "balance_classes": balance_classes,
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

    if checkpoint_epoch is not None:
        print('Saving checkpoints to', save_path)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    if model_name == 'runet':
        model = RUNet(1, 2)

    else:
        raise ValueError

    model = model.to(device)

    base_transform = CustomTransform(device, n_classes=2, mean=[0.49932378], std=[0.18392171])

    train_data = SyntheticDataset('data/SynthPokemonSegmentation/train', transform=base_transform)
    test_data = SyntheticDataset('data/SynthPokemonSegmentation/test', transform=base_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError

    if balance_classes is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(balance_classes).to(device) / sum(balance_classes))

    evaluations = Evaluations([
        'accuracy',
        'precision',
        'recall',
        'f1_score',
    ])

    loop = tqdm(range(epochs), desc='Main')
    for epoch in loop:
        log_data = {}

        train_log = train(model, train_loader, evaluations, device, criterion, optimizer)
        test_log = test(model, test_loader, evaluations, device)

        log_data.update({f"train_{k}": v for k, v in train_log.items()})
        log_data.update({f"test_{k}": v for k, v in test_log.items()})

        loop.set_postfix(log_data)

        if save_path and checkpoint_epoch is not None and (epoch % checkpoint_epoch) == (checkpoint_epoch - 1):
            save_checkpoint(model, optimizer, epoch, save_path, f"{model_name}-checkpoint-{epoch:04d}.pth")

        wandb.log(log_data)

    if save_path:
        torch.save(model.state_dict(), save_path / f"{model_name}.pth")

    wandb.finish()


if __name__ == "__main__":
    argh.dispatch_command(main)
