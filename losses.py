import math
import torch.nn as nn

from torch.nn import functional as F


def evaluate_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_targets = (targets == 1).float()
    jaccard_outputs = F.sigmoid(outputs)

    intersection = (jaccard_outputs * jaccard_targets).sum()
    union = jaccard_outputs.sum() + jaccard_targets.sum()

    jaccard = (intersection + eps) / (union - intersection + eps)

    return jaccard


class SoftJaccardBCEWithLogitsLoss:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """
    def __init__(self, jaccard_weight=0.0):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.jacc_weight = jaccard_weight

    def __call__(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)

        jaccard = evaluate_jaccard(outputs, targets)
        log_jaccard = math.log(jaccard)

        loss = bce_loss - self.jacc_weight * log_jaccard

        return loss
