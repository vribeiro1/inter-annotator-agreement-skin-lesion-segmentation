import os
import funcy
import math
import itertools

from torch.utils.data import Dataset
from torchvision.transforms import (Compose, ToTensor, Normalize, Resize)
from typing import Callable, List, Tuple
from PIL import Image


class SkinLesionSegmentationDataset(Dataset):
    def __init__(self, fpath: str, augmentation: List=None, input_preprocess: Callable=None, target_preprocess: Callable=None,
                 with_targets: bool=True, shape: Tuple=(256, 256)):
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Could not find dataset file: '{}'".format(fpath))
        self.with_targets = with_targets
        self.size = shape

        if input_preprocess is None:
            input_preprocess = [
                Resize(size=self.size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            input_preprocess = [
                Resize(size=self.size),
                input_preprocess,
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        if target_preprocess is None:
            target_preprocess = [
                Resize(size=self.size),
                ToTensor()
            ]
        else:
            target_preprocess = [
                Resize(size=self.size),
                target_preprocess,
                ToTensor()
            ]

        if not augmentation:
            augmentation = []
        n_augmentation = math.factorial(len(augmentation)) if len(augmentation) > 0 else 0
        augmentation_combinations = list(itertools.product([0, 1], repeat=n_augmentation))

        self.input_transform = Compose(input_preprocess)
        self.target_transform = Compose(target_preprocess)

        self.augmentation = augmentation

        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))
            if self.with_targets:
                data = [(input.strip(), target.strip())
                        for input, target in funcy.walk(lambda l: l.split(" "), lines)]
            else:
                data = [(input.strip(), None) for input in lines]

        self.data = [(d, transform_list) for transform_list in augmentation_combinations for d in data]

    @staticmethod
    def _load_input_image(fpath):
        img = Image.open(fpath).convert("RGB")
        return img

    @staticmethod
    def _load_target_image(fpath):
        img = Image.open(fpath).convert("P")
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (input_fpath, target_fpath), aug_bins = self.data[item]
        augmentation = [aug for aug, valid in zip(self.augmentation, aug_bins) if bool(valid)]
        aug_compose = Compose(augmentation)

        input_img = self._load_input_image(input_fpath)
        input_img = aug_compose(input_img)
        input_img = self.input_transform(input_img)

        target_img = None
        if target_fpath is not None:
            target_img = self._load_target_image(target_fpath)
            target_img = aug_compose(target_img)
            target_img = self.target_transform(target_img)

        fname = os.path.basename(input_fpath).split(".")[0]

        return input_img, target_img, fname


if __name__ == "__main__":
    fpath = "/Users/vribeiro/Documents/isic/train.txt"

    dataset = SkinLesionSegmentationDataset(fpath)
    for input_img, target_img, fname in dataset:
        print(input_img.size(), target_img.size())
