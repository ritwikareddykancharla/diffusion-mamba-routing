import torch

def sinkhorn(logits, iters=20):
    P = torch.exp(logits)
    for _ in range(iters):
        P = P / P.sum(dim=-1, keepdim=True)
        P = P / P.sum(dim=-2, keepdim=True)
    return P
