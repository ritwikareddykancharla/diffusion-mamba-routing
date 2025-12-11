# Minimal CVRP training loop
import torch
import torch.optim as optim
from core.diffusion import DiffusionDenoiser
from core.mamba_blocks import MambaBlock
from core.utils import pairwise_distance
from core.constraints import capacity_penalty

def train():
    B, N = 4, 10
    coords = torch.rand(B, N, 2)
    demands = torch.rand(B, N)
    capacity = torch.tensor([5.0]*B)

    model = DiffusionDenoiser(dim=2)
    mamba = MambaBlock(dim=2)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(50):
        feats = mamba(coords)
        noisy_adj = torch.rand(B, N, N)
        soft_adj = model(noisy_adj, feats)

        dist = pairwise_distance(coords)
        cost = (soft_adj * dist).sum()

        cap = capacity_penalty(soft_adj, demands, capacity).mean()

        loss = cost + cap

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(step, loss.item())

if __name__ == "__main__":
    train()
