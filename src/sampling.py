import torch
import torch.nn as nn


### TODO: add classifier-free guidance
### TODO: add restart sampling


def pushforward_sampling(model: nn.Module, timesteps: torch.tensor, noise:torch.tensor, omega: torch.tensor = None )->torch.tensor:
    num_steps = timesteps.shape
    x = noise
    for i in range(1, num_steps):
        if omega:
            x = model(x, timesteps[i], timesteps[i-1], omega )
        else:
            x = model(x, timesteps[i], timesteps[i - 1])
    return x