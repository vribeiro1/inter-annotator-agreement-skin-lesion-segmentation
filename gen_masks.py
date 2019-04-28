import argparse
import os
import torch
import numpy as np
import random

from skimage.io import imsave
from skimage.transform import resize
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm

from dataset import SkinLesionSegmentationDataset


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def main(model_path, data_path, save_to=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location="cpu").to(device)
    dataset = SkinLesionSegmentationDataset(data_path, with_targets=False)
    dataloader = data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, worker_init_fn=set_seeds)

    model.eval()
    progress_bar = tqdm(dataloader, desc="Generating images")
    for i, (inputs, targets, fnames, (widths, heights)) in enumerate(progress_bar):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = F.sigmoid(model(inputs))

        for output, fname, width, height in zip(outputs, fnames, widths, heights):
            output_img = output.float().squeeze().numpy()
            output_img[output_img < 0.5] = 0.0
            output_img[output_img >= 0.5] = 1.0
            output_img = resize(output_img, (height, width))
            imsave(os.path.join(save_to, fname + ".png"), output_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest="model_path")
    parser.add_argument("--save-to", dest="save_to")
    parser.add_argument("--data-path", dest="data_path")
    args = parser.parse_args()

    if not os.path.isdir(args.save_to):
        os.makedirs(args.save_to)

    main(args.model_path, args.data_path, args.save_to)
