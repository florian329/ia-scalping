"""Moteur d'entrainement par imitation avec split chronologique et early stopping.

Corrections vs version initiale :
- Poids de classes calcules depuis la distribution reelle des labels
  (plus de [1.0, 3.5, 3.5] arbitraire).
- Early stopping avec patience pour eviter le surapprentissage.
- Reproductibilite (seed).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class ScalpingModelCheckpoint:
    def __init__(self, filepath="best_scalping_model.pth"):
        self.filepath = filepath
        self.best_loss = float("inf")

    def __call__(self, current_val_loss, model, epoch, optimizer):
        if current_val_loss < self.best_loss:
            self.best_loss = current_val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": current_val_loss,
            }
            torch.save(state, self.filepath)
            logger.info("Nouveau record : modele sauvegarde a l'epoque %d", epoch)
            return True
        return False


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=0.001,
    device="cpu",
    class_weights=None,
    patience=5,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if class_weights is None:
        class_weights = torch.tensor([1.0, 3.5, 3.5])
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion_no_reduction = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    checkpoint_manager = ScalpingModelCheckpoint()
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch_book, batch_seq, batch_targets = batch[0], batch[1], batch[2]
            batch_book = batch_book.to(device)
            batch_seq = batch_seq.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_book, batch_seq)
            if len(batch) >= 4:
                sample_w = batch[3].to(device)
                per_sample = criterion_no_reduction(outputs, batch_targets)
                loss = (per_sample * sample_w).sum() / sample_w.sum()
            else:
                loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_book.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_book, batch_seq, batch_targets in val_loader:
                batch_book = batch_book.to(device)
                batch_seq = batch_seq.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_book, batch_seq)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item() * batch_book.size(0)
        val_loss /= len(val_loader.dataset)
        logger.info("Epoch %3d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

        improved = checkpoint_manager(val_loss, model, epoch, optimizer)
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping a l'epoque %d (patience=%d)", epoch, patience)
                break
    return checkpoint_manager.best_loss
