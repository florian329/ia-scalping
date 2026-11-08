"""Calibration des probabilites par temperature scaling.

Un reseau de classification produit des probabilites sur-confiantes : un
seuil de decision a 0.5 sur des probas mal calibrees ne veut rien dire. Le
temperature scaling divise les logits par T appris sur la validation (NLL
minimisation, un seul parametre) : l'ordre des classes est preserve, mais
les probabilites deviennent interpretables.
"""

import numpy as np
import torch
import torch.nn as nn


def fit_temperature(model, val_loader, device="cpu", max_iter=100, lr=0.1):
    """Apprend T qui minimise la NLL sur le jeu de validation."""
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            book, seq, targets = batch[0], batch[1], batch[2]
            logits = model(book.to(device), seq.to(device))
            logits_list.append(logits.cpu())
            targets_list.append(targets)
    if not logits_list:
        return 1.0
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature, min=0.05).item())


def expected_calibration_error(probs, targets, n_bins=10):
    """ECE : ecart moyen entre confiance et exactitude par bin de probabilite."""
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == targets
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidences)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)
