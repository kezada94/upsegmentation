from pathlib import Path

import argh
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import parallel_average


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

        image = Image.open(str(image))
        target = Image.open(str(target))

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
