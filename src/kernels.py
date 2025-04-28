import torch


def l2_kernel(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return -torch.linalg.norm(x - y)


def psudo_huber(x: torch.tensor, y: torch.tensor, c: int = 1.0) -> torch.tensor:
    return c - torch.linalg.norm(x - y + c)


def exponential_kernel(
    x: torch.Tensor, y: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    return torch.exp(-beta * torch.linalg.norm(x - y))


def cosine_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.linalg.norm(x)
    y_norm = torch.linalg.norm(y)
    return torch.dot(x.flatten(), y.flatten()) / (x_norm * y_norm + 1e-8)


def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.flatten(), y.flatten())
