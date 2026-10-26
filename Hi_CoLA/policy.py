import torch
import torch.nn as nn


class Behavioural_v2(nn.Module):
    """Behavior policy: maps a 35-d FRT observation to 4 independent detour
    probabilities (one per fixed stop decision point)."""

    def __init__(self, input_dim=35, hidden_dim=64, output_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
