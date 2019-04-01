import os
import numpy as np
import torch
import random

from collections import OrderedDict
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import ToPILImage, ToTensor
from sacred import Experiment
from sacred.observers import FileStorageObserver
from skimage.morphology import square

from dataset import SkinLesionSegmentationDataset
from losses import SoftJaccardBCEWithLogitsLoss, evaluate_jaccard, evaluate_dice
from model.deeplab import DeepLab
from summary_writer import SummaryWriter
from transforms.target import Opening, ConvexHull
from transforms.input import GaussianNoise, EnhanceBrightness, EnhanceContrast, EnhanceColor, EnhanceSharpness, ColorGradient

base_path = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create("results")
ex.observers.append(fs_observer)


def postprocess_batch(transform_fn, batch_tensor):
    transformed_batch = []
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    for tensor in batch_tensor:
        pil_img = to_pil(tensor)
        img = to_tensor(transform_fn(pil_img))
        transformed_batch.append(img)

    return torch.stack(tuple(transformed_batch))


available_transforms = {
    "original": lambda x: x,
    "opening": Opening(square, 5),
    "convex_hull": ConvexHull()
}


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def run_epoch(phase, epoch, model, dataloader, postprocess, optimizer, criterion, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    jaccards = []
    dices = []
    for i, (inputs, targets, fname) in enumerate(progress_bar):
        inputs = Variable(inputs, requires_grad=True).to(device)
        targets = Variable(targets, requires_grad=True).to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            # outputs = postprocess_batch(postprocess, outputs)

            loss = criterion(outputs, targets)
            jaccard = evaluate_jaccard(outputs, targets)
            dice = evaluate_dice(jaccard.item())

            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            jaccards.append(jaccard.item())
            dices.append(dice)
            progress_bar.set_postfix(OrderedDict({"{} loss".format(phase): np.mean(losses),
                                                  "{} jaccard".format(phase): np.mean(jaccards),
                                                  "{} dice".format(phase): np.mean(dices)}))

    mean_loss = np.mean(losses)
    mean_jacc = np.mean(jaccards)
    mean_dice = np.mean(dices)

    loss_tag = "{}.loss".format(phase)
    jacc_tag = "{}.jaccard".format(phase)
    dice_tag = "{}.dice".format(phase)

    writer.add_scalar(loss_tag, mean_loss, epoch)
    writer.add_scalar(jacc_tag, mean_jacc, epoch)
    writer.add_scalar(dice_tag, mean_dice, epoch)

    info = {"loss": mean_loss,
            "jaccard": mean_jacc,
            "dice": mean_dice}

    return info


@ex.automain
def main(batch_size, n_epochs, lr, decay, train_fpath, val_fpath, train_preprocess, val_preprocess, postprocess, _run):
    assert train_preprocess in available_transforms, "Train pre-process '{}' is not available. Available functions are: '{}'".format(train_preprocess, list(available_transforms.keys()))
    assert val_preprocess in available_transforms, "Validation pre-process '{}' is not available. Available functions are: '{}'".format(val_preprocess, list(available_transforms.keys()))
    assert postprocess in available_transforms, "Post-process '{}' is not available. Available functions are: '{}'".format(postprocess, list(available_transforms.keys()))

    train_preprocess_fn = available_transforms[train_preprocess]
    val_preprocess_fn = available_transforms[val_preprocess]
    postprocess_fn = available_transforms[postprocess]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(base_path, "runs", "experiment-{}".format(_run._id)))
    model_path = os.path.join(fs_observer.dir, "best_model.pth")

    outputs_path = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    model = DeepLab(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)

    augmentations = [
        GaussianNoise(0, 4),
        EnhanceBrightness(0.5, 0.1),
        EnhanceContrast(0.5, 0.1),
        EnhanceColor(0.5, 0.1),
        EnhanceSharpness(1.5, 0.1),
        ColorGradient()
    ]

    train_dataset = SkinLesionSegmentationDataset(train_fpath, augmentations=augmentations, target_preprocess=train_preprocess_fn)
    val_dataset = SkinLesionSegmentationDataset(val_fpath, target_preprocess=val_preprocess_fn)

    dataloaders = {
        "train": data.DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 shuffle=True,
                                 worker_init_fn=set_seeds),
        "validation": data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      num_workers=8,
                                      shuffle=False,
                                      worker_init_fn=set_seeds)
    }

    info = {}
    epochs = range(1, n_epochs + 1)
    best_jacc = np.inf

    for epoch in epochs:
        info["train"] = run_epoch("train", epoch, model, dataloaders["train"], postprocess_fn, optimizer, loss_fn, writer)
        info["validation"] = run_epoch("validation", epoch, model, dataloaders["validation"], postprocess_fn, optimizer, loss_fn, writer)

        if epoch == 1 or epoch % 10 == 0:
            writer.commit()

        if info["validation"]["loss"] < best_jacc:
            best_jacc = info["validation"]["jaccard"]
            torch.save(model, model_path)

        if epoch % 25 == 0:
            lr = max(lr * decay, 0.00000001)
            optimizer = optim.Adam(model.parameters(), lr=lr)
