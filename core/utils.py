import torch

def pairwise_distance(coords):
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return torch.norm(diff, dim=-1)
