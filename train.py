import os
import numpy as np
import torch

from collections import OrderedDict
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from sacred import Experiment
from sacred.observers import FileStorageObserver
from random import random

from dataset import SkinLesionSegmentationDataset
from losses import SoftJaccardBCEWithLogitsLoss, evaluate_jaccard
from model.deeplab import DeepLab
from summary_writer import SummaryWriter

base_path = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create("results")
ex.observers.append(fs_observer)


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    jaccards = []
    for i, (inputs, targets, fname) in enumerate(progress_bar):
        inputs = Variable(inputs).to(device)
        targets = Variable(targets).to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            jaccard = evaluate_jaccard(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            jaccards.append(jaccard.item())
            progress_bar.set_postfix(OrderedDict({"{} loss".format(phase): np.mean(losses),
                                                  "{} jaccard".format(phase): np.mean(jaccards)}))

    mean_loss = np.mean(losses)
    mean_jacc = np.mean(jaccards)

    loss_tag = "{}.loss".format(phase)
    jacc_tag = "{}.jaccard".format(phase)

    writer.add_scalar(loss_tag, mean_loss, epoch)
    writer.add_scalar(jacc_tag, mean_jacc, epoch)

    info = {"loss": mean_loss,
            "jaccard": mean_jacc}

    return info


@ex.automain
def main(batch_size, n_epochs, lr, decay, train_fpath, val_fpath, _run):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(os.path.join(base_path, "runs", "experiment-{}".format(_run._id)))
    model_path = os.path.join(fs_observer.dir, "best_model.pth")

    outputs_path = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    model = DeepLab(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)

    train_dataset = SkinLesionSegmentationDataset(train_fpath)
    val_dataset = SkinLesionSegmentationDataset(val_fpath)

    dataloaders = {
        "train": data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, worker_init_fn=set_seeds),
        "validation": data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, worker_init_fn=set_seeds)
    }

    info = {}
    epochs = range(1, n_epochs + 1)
    best_loss = np.inf

    for epoch in epochs:
        run_epoch("train", epoch, model, dataloaders["train"], optimizer, loss_fn, writer)
        run_epoch("validation", epoch, model, dataloaders["validation"], optimizer, loss_fn, writer)

        if info["validation"]["loss"] < best_loss:
            best_loss = info["validation"]["loss"]
            torch.save(model, model_path)

        if epoch % 25 == 0:
            lr = max(lr * decay, 0.00000001)
            optimizer = optim.Adam(model.parameters(), lr=lr)
