from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = Path(path)
        self.transform = transform

        self.targets = list(self.path.glob('*_target.png'))
        self.images = [(i.parent / (i.stem[:-7] + '.png')) for i in self.targets]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        image = Image.open(str(image))
        target = Image.open(str(target))

        image = np.asarray(image)
        target = np.asarray(target)

        image = image[:, :, np.newaxis]
        target = np.stack([target, 255 - target], axis=2)

        image = image / 255
        target = target / 255

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image.float(), target.float()
