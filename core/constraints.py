# Soft MILP constraints for CVRP
import torch

def capacity_penalty(soft_adj, demands, capacity):
    incoming = soft_adj.sum(dim=1)
    load = (incoming * demands).sum(dim=1)
    return torch.relu(load - capacity)
