"""Metriques de classification et matrice de confusion (sans dependance seaborn)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Attente (0)", "Scalp Long (1)", "Scalp Short (2)"]


def evaluate_and_plot(y_true, y_pred, save_path="confusion_matrix.png"):
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(3), CLASS_NAMES, rotation=30)
    plt.yticks(range(3), CLASS_NAMES)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Vraies decisions")
    plt.xlabel("Predictions IA")
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matrice de confusion sauvegardee sous '{save_path}'")
    return cm


def per_class_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = y_true == c
        out[name] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    return out
