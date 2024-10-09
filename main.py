import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import argh
import wandb
import numpy as np
import torch.optim
import torch.nn as nn

from tqdm import tqdm
from PIL import ImageOps, Image
from skimage.transform import resize
from torch.utils.data import DataLoader

from models import *
from evaluations import *
from dataset import FLHDataset
from utils import FloatAction, labels_to_one_hot, one_hot_to_labels
from loss import dice_loss

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
          device: Union[torch.device, str],
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer) -> Dict[str, Any]:

    evaluation.reset()
    log_data = {'loss': 0.0}
    loop = tqdm(loader, desc='Train')
    with torch.set_grad_enabled(True):
        model.train()
        for batch_idx, (x, yt) in enumerate(loop):
            # print ("yt", yt.shape)
            one_hot_yt = labels_to_one_hot(yt).to(torch.float32)
            # print ("one_hot_yt", one_hot_yt.shape)
            x = x.to(device)
            yt = yt.to(device)
            one_hot_yt = one_hot_yt.to(device)

            optimizer.zero_grad()
            one_hot_yp = model(x)
            # print(one_hot_yp.shape)
            loss = criterion(one_hot_yp, one_hot_yt)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            log_data['loss'] = batch_loss

            yp = one_hot_to_labels(one_hot_yp)
            log_data.update(evaluation(yt, yp))
            loop.set_postfix(log_data)
    return log_data


def test(model: nn.Module,
         loader: DataLoader,
         evaluation: Evaluations,
         device: Union[torch.device, str],
         save_folder: Path) -> Dict[str, Any]:
    class_labels = {0: "background", 1: "border"}
    evaluation.reset()
    log_data = {}
    predictions = []
    loop = tqdm(loader, desc='Test')
    save_folder.mkdir(parents=True, exist_ok=True)
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, yt) in enumerate(loop):
            x = x.to(device)
            yt = yt.to(device)
            yp = (model(x))
            log_data.update(evaluation(yt, yp))
            for img_idx in range(len(x)):
                yp_img = yp[img_idx].detach().cpu().numpy()
                yp_img = (yp_img * 255).astype(np.uint8)[0]
                # yp_img = one_hot_to_labels(yp_img)
                # print (yp_img.shape)
                # Ensure yt_img has a valid shape for PIL
                # if yp_img.ndim == 3 and yp_img.shape[0] == 1:
                #     yp_img = yp_img.squeeze(0)
                # elif yp_img.ndim == 2:
                #     yp_img = yp_img
                # else:
                #     raise ValueError(f"Unexpected shape for yp_img: {yp_img.shape}")
                
                # Save the prediction image
                pred_image = Image.fromarray(yp_img)
                pred_image.save(save_folder / f"prediction_{batch_idx}_{img_idx}.png")

            loop.set_postfix(log_data)
    log_data['prediction'] = predictions
    return log_data

#
class CustomTransform:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, x, y):
        # Normalization
        x = (x - self.mean) / self.std
        return x, y


@argh.arg("epochs", type=int)
@argh.arg("model-name", type=str, choices=['runet'])
@argh.arg("block-type", type=str, choices=['resnet', 'mobilenetv2'])
@argh.arg("dataset-path", type=Path)
@argh.arg("in-size", type=int)
@argh.arg("scale", type=int)
@argh.arg("--mean", type=float, nargs='+', action=FloatAction, default=0)
@argh.arg("--std", type=float, nargs='+', action=FloatAction, default=1)
@argh.arg("--optimizer", type=str, default='adam')
@argh.arg("--batch-size", type=int, default=64)
@argh.arg("--learning-rate", type=float, default=1e-6)
@argh.arg("--balance-classes", type=float, nargs=2, default=None)
@argh.arg("--use-cuda", default=True)
@argh.arg("--num-workers", type=int, default=4)
@argh.arg("--checkpoint-epoch", type=int, default=None)
@argh.arg("--save-path", type=Path, default=None)
@argh.arg("--seed", type=int, default=None)
def main(epochs: int,
         model_name: str,
         block_type: str,
         dataset_path: Path,
         in_size: int,
         scale: int,
         mean: Union[float, Tuple[float, float, float]] = 0,
         std: Union[float, Tuple[float, float, float]] = 1,
         optimizer: str = 'adam',
         batch_size=64,
         learning_rate=1e-4,
         balance_classes: Tuple[float, float] = None,
         use_cuda: bool = True,
         num_workers=4,
         checkpoint_epoch: int = None,
         save_path: Path = None,
         seed: int = None):

    block_type = block_type.lower()


    device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"


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
        model = RUNet(1, 1, scale=scale, down_block=block_type)

    else:
        raise ValueError()

    model = model.to(device)

    base_transform = CustomTransform(mean=np.float32(mean), std=np.float32(std))

    dataset = FLHDataset(dataset_path, transform=base_transform, contour=False, resize=in_size)
    # perform a 80-20 split of the dataset keeping the same distribution of classes
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # show some examples using matplotlib
    # show some examples using matplotlib
    import matplotlib.pyplot as plt

    def show_examples(loader, num_examples=4):
        model.eval()
        examples = next(iter(loader))
        x, yt = examples
        with torch.no_grad():
            yp = (model(x.to(device)))
            # yp = one_hot_to_labels(yp)
            print(yp)
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, num_examples * 5))
        for i in range(num_examples):
            print(yp[i].shape)
            axes[i, 0].imshow(np.transpose(x[i].numpy(), (1, 2, 0)))
            axes[i, 0].set_title("Input Image")
            axes[i, 1].imshow(np.transpose(yt[i].numpy(), (1, 2, 0)), cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].imshow(yp[i].cpu().numpy().squeeze(), cmap='gray')
            axes[i, 2].set_title("Prediction")
        plt.show()
        model.train()

    show_examples(test_loader)


    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError

    if balance_classes is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(balance_classes).to(device) / sum(balance_classes))
    # criterion = dice_loss
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
        test_log = test(model, test_loader, evaluations, device, save_path / f"predictions_epoch_{epoch}")

        log_data.update({f"train_{k}": v for k, v in train_log.items()})
        log_data.update({f"test_{k}": v for k, v in test_log.items()})

        loop.set_postfix(log_data)

        if save_path and checkpoint_epoch is not None and (epoch % checkpoint_epoch) == (checkpoint_epoch - 1):
            save_checkpoint(model, optimizer, epoch, save_path, f"{model_name}-checkpoint-{epoch:04d}.pth")


    if save_path:
        torch.save(model.state_dict(), save_path / f"{model_name}.pth")



if __name__ == "__main__":
    argh.dispatch_command(main)