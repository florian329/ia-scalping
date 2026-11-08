"""Modele hybride MLP + LSTM a deux branches (carnet d'ordres + price action).

Correction vs version initiale : BatchNorm1d remplace par LayerNorm.
BatchNorm sur la branche carnet est dangereux en production (statistiques de
batch non stationnaires, comportement incoherent en inference batch=1).
"""

import torch
import torch.nn as nn


class ScalpingMultimodalNet(nn.Module):
    def __init__(self, order_book_dim=4, sequence_dim=4, hidden_dim=64):
        super().__init__()
        self.order_book_branch = nn.Sequential(
            nn.Linear(order_book_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.lstm_branch = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.lstm_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, order_book_data, sequential_data):
        out_book = self.order_book_branch(order_book_data)
        lstm_out, _ = self.lstm_branch(sequential_data)
        out_sequence = self.lstm_fc(lstm_out[:, -1, :])
        combined = torch.cat((out_book, out_sequence), dim=1)
        return self.fusion_layer(combined)


def predict_action(model, book_t, seq_window, device="cpu"):
    """Inférence single-tick : retourne l'action (0/1/2) et ses probabilites."""
    model.eval()
    with torch.no_grad():
        book = torch.tensor(book_t, dtype=torch.float32).unsqueeze(0).to(device)
        seq = torch.tensor(seq_window, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(book, seq)
        probs = torch.softmax(logits, dim=1)
        action = int(torch.argmax(probs, dim=1).item())
    return action, probs.squeeze(0).cpu().numpy()
