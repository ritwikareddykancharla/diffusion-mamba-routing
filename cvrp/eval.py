# Minimal evaluation
import torch
from core.utils import pairwise_distance

def evaluate(model, coords):
    dist = pairwise_distance(coords)
    N = coords.size(1)
    feats = model.mamba(coords)
    adj = model.diffuser(torch.rand(1, N, N), feats)
    cost = (adj * dist).sum()
    return cost.item()
