import torch


def l2_kernel(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return -torch.linalg.norm(x - y)


def psudo_huber(x: torch.tensor, y: torch.tensor, c: int = 1.0) -> torch.tensor:
    return c - torch.linalg.norm(x - y + c)
