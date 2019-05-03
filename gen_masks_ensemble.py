import argparse
import cv2
import os
import torch
import numpy as np
import random

from skimage.transform import resize
from torch.utils import data
from tqdm import tqdm

from dataset import SkinLesionSegmentationDataset


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def main(model_paths, data_path, save_to=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [torch.load(model_path, map_location="cpu").to(device) for model_path in model_paths]
    dataset = SkinLesionSegmentationDataset(data_path, with_targets=False)
    dataloader = data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, worker_init_fn=set_seeds)

    progress_bar = tqdm(dataloader, desc="Generating images")
    for i, (inputs, targets, fnames, (widths, heights)) in enumerate(progress_bar):
        all_outputs = []
        for model in models:
            model.eval()

            with torch.no_grad():
                outputs = torch.sigmoid(model(inputs))
                all_outputs.append(outputs)

        all_outputs = zip(*all_outputs)
        summed_outputs = []
        for outputs in all_outputs:
            summed_outputs.append(sum(outputs) / len(models))

        for output, fname, width, height in zip(summed_outputs, fnames, widths, heights):
            output_img = output.float().squeeze().numpy()
            output_img = resize(output_img, (height, width))
            output_img[output_img < 0.5] = 0.0
            output_img[output_img >= 0.5] = 255.0

            cv2.imwrite(os.path.join(save_to, fname + "_segmentation.png"), output_img.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", nargs='+', dest="model_paths")
    parser.add_argument("--save-to", dest="save_to")
    parser.add_argument("--data-path", dest="data_path")
    args = parser.parse_args()

    if not os.path.isdir(args.save_to):
        os.makedirs(args.save_to)

    main(args.model_paths, args.data_path, args.save_to)
