import argparse
import os
import torch
import numpy as np
import random
import json

from collections import OrderedDict
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from dataset import SkinLesionSegmentationDataset
from losses import SoftJaccardBCEWithLogitsLoss, evaluate_jaccard, evaluate_dice

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def run_test(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_bar = tqdm(dataloader, desc="Testing")

    model.eval()

    losses = []
    jaccards = []
    dices = []
    for i, (inputs, targets, fname, (_, _)) in enumerate(progress_bar):
        inputs = Variable(inputs, requires_grad=True).to(device)
        targets = Variable(targets, requires_grad=True).to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            jaccard = evaluate_jaccard(outputs, targets)
            dice = evaluate_dice(jaccard.item())

            losses.append(loss.item())
            jaccards.append(jaccard.item())
            dices.append(dice)
            progress_bar.set_postfix(OrderedDict({"loss": np.mean(losses),
                                                  "jaccard": np.mean(jaccards),
                                                  "dice": np.mean(dices)}))

    mean_loss = np.mean(losses)
    mean_jacc = np.mean(jaccards)
    mean_dice = np.mean(dices)

    info = {"loss": mean_loss,
            "jaccard": mean_jacc,
            "dice": mean_dice}

    return info


def main(data_name, fpath, model_path, save_to=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkinLesionSegmentationDataset(fpath)
    dataloader = data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, worker_init_fn=set_seeds)
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)
    model = torch.load(model_path).to(device)

    info = run_test(model, dataloader, loss_fn)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    with open(os.path.join(save_to, "test_{}.json".format(data_name)), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest="model_path")
    parser.add_argument("--save-to", dest="save_to")
    args = parser.parse_args()

    test_dermofit_path = os.path.join(BASE_PATH, "data", "test_dermofit.txt")
    test_isic_titans_path = os.path.join(BASE_PATH, "data", "isic2017_test.txt")
    test_ph2_path = os.path.join(BASE_PATH, "data", "test_ph2.txt")

    # main("dermofit", test_dermofit_path, args.model_path, args.save_to)
    main("isic_2017", test_isic_titans_path, args.model_path, args.save_to)
    main("ph2", test_ph2_path, args.model_path, args.save_to)


# CUDA_VISIBLE_DEVICES=GPU-7bbed26b python3 test.py --model-path /workspace/skin/results/1/best_model.pth --save-to /workspace/skin/results/1
# CUDA_VISIBLE_DEVICES=GPU-7bbed26b python3 test.py --model-path /workspace/skin/results/2/best_model.pth --save-to /workspace/skin/results/2
# CUDA_VISIBLE_DEVICES=GPU-7bbed26b python3 test.py --model-path /workspace/skin/results/3/best_model.pth --save-to /workspace/skin/results/3
# CUDA_VISIBLE_DEVICES=GPU-7bbed26b python3 test.py --model-path /workspace/skin/results/4/best_model.pth --save-to /workspace/skin/results/4
