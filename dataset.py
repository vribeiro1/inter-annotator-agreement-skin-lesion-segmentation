import os
import funcy

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Resize
from typing import Callable, List, Tuple
from PIL import Image


class SkinLesionSegmentationDataset(Dataset):
    def __init__(self, fpath: str, augmentations: List=None, input_preprocess: Callable=None, target_preprocess: Callable=None,
                 with_targets: bool=True, shape: Tuple=(256, 256)):
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Could not find dataset file: '{}'".format(fpath))
        self.with_targets = with_targets
        self.size = shape

        if augmentations:
            augmentations = [lambda x: x] + augmentations
        else:
            augmentations = [lambda x: x]

        self.resize = Resize(size=self.size)
        self.to_tensor = ToTensor()
        self.input_preprocess = input_preprocess
        self.target_preprocess = target_preprocess

        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))
            if self.with_targets:
                data = [(input.strip(), target.strip())
                        for input, target in funcy.walk(lambda l: l.split(" "), lines)]
            else:
                data = [(input.strip(), None) for input in lines]

        self.data = [(d, augmentation) for augmentation in augmentations for d in data]

    @staticmethod
    def _load_input_image(fpath: str):
        img = Image.open(fpath).convert("RGB")
        return img

    @staticmethod
    def _load_target_image(fpath: str):
        img = Image.open(fpath).convert("L")
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (input_fpath, target_fpath), augmentation = self.data[item]

        input_img = self._load_input_image(input_fpath)
        input_img = self.resize(input_img)

        if self.input_preprocess is not None:
            input_img = self.input_preprocess(input_img)

        input_img = augmentation(input_img)
        input_img = self.to_tensor(input_img)

        target_img = None
        if target_fpath is not None:
            target_img = self._load_target_image(target_fpath)
            target_img = self.resize(target_img)

            if self.target_preprocess is not None:
                target_img = self.target_preprocess(target_img)

            target_img = self.to_tensor(target_img)

        fname = os.path.basename(input_fpath).split(".")[0]

        return input_img, target_img, fname


if __name__ == "__main__":
    from transforms.input import GaussianNoise, EnhanceBrightness, EnhanceContrast, EnhanceColor, EnhanceSharpness, ColorGradient
    from transforms.target import Opening
    from skimage.morphology import square

    fpath = "/Users/vribeiro/Documents/isic/train.txt"

    to_pil = ToPILImage()

    target_preprocess = Opening(square, 5)
    augmentations = [
        GaussianNoise(0, 4),
        EnhanceBrightness(0.5, 0.1),
        EnhanceContrast(0.5, 0.1),
        EnhanceColor(0.5, 0.1),
        EnhanceSharpness(1.5, 0.1),
        ColorGradient()
    ]
    dataset = SkinLesionSegmentationDataset(fpath, augmentations=augmentations, target_preprocess=target_preprocess)
    print(len(dataset))
    for input_img, target_img, fname in dataset:
        print(input_img.size(), target_img.size())
