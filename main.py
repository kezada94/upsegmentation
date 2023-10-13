import csv
import argh
from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from unet import UNet
from dataset import SyntheticDataset


def count_true_positives(yp, yt):
    return 0


def count_true_negatives(yp, yt):
    return 0


def count_false_positives(yp, yt):
    return 0


def count_false_negatives(yp, yt):
    return 0


@argh.arg("epochs", type=int)
@argh.arg("--use-cuda", default=True)
@argh.arg("--batch-size", type=int, default=64)
@argh.arg("--num-workers", type=int, default=4)
@argh.arg("--checkpoint-epoch", type=int, default=10)
def train(epochs: int, use_cuda: bool = True, batch_size=64, num_workers=4, checkpoint_epoch: int = 10):
    use_cuda = use_cuda and torch.cuda.is_available()

    model = UNet(1, 2)

    data = SyntheticDataset('data/generated/png', transforms.ToTensor())

    train_data, test_data = torch.utils.data.random_split(data, [800, 200])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    evaluation = {
        'true_positives': count_true_positives,
        'true_negatives': count_true_negatives,
        'false_positives': count_false_positives,
        'false_negatives': count_false_negatives,
    }

    if use_cuda:
        model = model.cuda()

    learning_curve = []

    epoch_loop = tqdm(range(epochs), position=0, desc='Epoch')
    for epoch in epoch_loop:
        learning_data = {'epoch': epoch}

        with torch.set_grad_enabled(True):
            learning_data['loss'] = 0.0

            train_loop = tqdm(train_loader, position=1, leave=False, desc='Train')

            model.train()
            for batch_idx, (x, yt) in enumerate(train_loop):
                if use_cuda:
                    x, yt = x.cuda(), yt.cuda()
                optimizer.zero_grad()
                yp = model(x)
                loss = criterion(yp, yt)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                learning_data['loss'] += batch_loss

                train_loop.set_postfix(epoch=epoch, batch=batch_idx, loss=batch_loss)

        with torch.set_grad_enabled(False):
            for ev_name in evaluation.keys():
                learning_data[ev_name] = 0

            test_loop = tqdm(test_loader, position=1, leave=False, desc='Test')

            model.eval()
            for batch_idx, (x, yt) in enumerate(test_loop):
                if use_cuda:
                    x, yt = x.cuda(), yt.cuda()
                yp = model(x)

                for ev_name, ev_fn in evaluation.items():
                    learning_data[ev_name] += ev_fn(yp, yt)

                test_loop.set_postfix(epoch=epoch, batch=batch_idx)

        learning_curve.append(learning_data)
        epoch_loop.set_postfix(learning_data)

        if (epoch % checkpoint_epoch) == (checkpoint_epoch - 1):
            torch.save(model.state_dict(), f"data/unet-checkpoint-{epoch:40d}.pth")

    torch.save(model.state_dict(), "data/unet.pth")

    with open("data/learning_curve.tsv", "w", newline="") as f:
        fields = list(learning_curve[0].keys())
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(learning_curve)


if __name__ == "__main__":
    # (?, 1, 112, 112) -> (?, 2, 186, 186)
    argh.dispatch_command(train)
