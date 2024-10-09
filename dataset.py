from pathlib import Path

import argh
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

import torch
from torchvision.transforms.functional import pil_to_tensor

class FLHDataset(Dataset):
    def __init__(self, path, transform=None, grayscale=True, contour=False, resize=None):
        super().__init__()
        self.path = Path(path)
        self.transform = transform
        self.grayscale = grayscale
        self.contour = contour
        if self.contour:
            self.prefix = 'contour'
        else:
            self.prefix = 'filled'
        self.resize = resize
        self.targets = []
        self.images = []

        for folder in self.path.glob('*TL_0212*'):
            print(f'Found folder: {folder}')
            if folder.is_dir():
                time = str(folder).split('TL')[0].split('/')[-1]
                print(f'Time: {time}')
                for cellFolder in folder.glob('AC_*'):
                    if cellFolder.is_dir():
                        print(f'Found cell folder: {cellFolder}')
                        for z in cellFolder.glob(self.prefix+'*.tif'):
                            self.targets.append(z)
                            # the z value contour_TL_0212_T26hpf_fill_c1_z43.tif
                            zval = z.stem.split('_')[-1].split('.')[0].replace('z', 'z0')
                            self.images.append(str(path)+'/' + time + "ch00_" + zval + ".tif")

                            print(f'Target: {z}, Image: {time + "ch00_" + zval + ".tif"}')
        print(f'Found {len(self.images)} images')

        # self.imgs = [None] * len(self.images)
        # self.targs = [None] * len(self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        image = Image.open(str(image)).convert('RGB')
        target = Image.open(str(target)).convert('RGB')
        

        if self.grayscale:
            image = ImageOps.grayscale(image)
            target = ImageOps.grayscale(target)

        if self.resize:
            # make the result binary
            image = image.resize((self.resize, self.resize), Image.NEAREST)
            target = target.resize((self.resize*2, self.resize*2), Image.NEAREST)

        image = pil_to_tensor(image).to(torch.float32)
        target = pil_to_tensor(target).to(torch.float32)

        image = image / 255.0
        target = 1-(target / 255.0)
        if self.transform:
            image, target = self.transform(image, target)
        #print the max and min of the target
        # print(f"Max: {torch.max(target)}, Min: {torch.min(target)}")
        return image, target




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
