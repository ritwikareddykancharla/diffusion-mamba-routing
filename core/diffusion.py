# Minimal diffusion denoiser
import torch
import torch.nn as nn
from .sinkhorn import sinkhorn

class DiffusionDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, noisy_adj, features):
        logits = self.mlp(features) @ self.mlp(features).transpose(-1, -2)
        soft_adj = sinkhorn(logits)
        return soft_adj
