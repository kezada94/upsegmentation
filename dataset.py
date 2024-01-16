from pathlib import Path

import argh
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

import torch
from torchvision.transforms.functional import pil_to_tensor


class SyntheticDataset(Dataset):
    def __init__(self, path, transform=None, grayscale=True):
        super().__init__()
        self.path = Path(path)
        self.transform = transform
        self.grayscale = grayscale

        self.targets = list(self.path.glob('*_target.png'))
        self.images = [(i.parent / (i.stem[:-7] + '_input.png')) for i in self.targets]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        image = Image.open(str(image)).convert('RGB')
        target = Image.open(str(target)).convert('RGB')

        if self.grayscale:
            image = ImageOps.grayscale(image)

        image = pil_to_tensor(image).to(torch.float32)
        target = pil_to_tensor(target).to(torch.float32)

        image = image / 255.0
        target = target / 255.0

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
