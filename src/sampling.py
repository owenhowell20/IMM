import os
import re
import json

import pickle
import functools
import numpy as np

import torch
import dnnlib

import torchvision.utils as vutils
import warnings

from omegaconf import OmegaConf
from torch_utils import misc
import hydra

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.



def generator_fn(*args, name='pushforward_generator_fn', **kwargs):
    return globals()[name](*args, **kwargs)


### pushforward sampling
@torch.no_grad()
def pushforward_generator_fn(net, latents, class_labels=None, discretization=None, mid_nt=None, num_steps=None,
                             cfg_scale=None, ):
    # Time step discretization.
    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps + 1, dtype=torch.float64, device=latents.device)
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        step_indices = torch.arange(num_steps + 1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (
                    nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt]
        t_steps = torch.tensor(
            [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
        )
        # t_0 = T, t_N = 0
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])

    # Sampling steps
    x = latents.to(torch.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale).to(
            torch.float64
        )

    return x


### restart sampling
@torch.no_grad()
def restart_generator_fn(net, latents, class_labels=None, discretization=None, mid_nt=None, num_steps=None,
                         cfg_scale=None):
    # Time step discretization.
    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps + 1, dtype=torch.float64, device=latents.device)[:-1]
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        step_indices = torch.arange(num_steps + 1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (
                    nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)[:-1]
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt]
        t_steps = torch.tensor(
            [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
        )
        # Sampling steps
    x = latents.to(torch.float64)

    for i, t_cur in enumerate(t_steps):

        x = net.cfg_forward(x, t_cur, torch.ones_like(t_cur) * net.eps, class_labels=class_labels,
                            cfg_scale=cfg_scale).to(
            torch.float64
        )

        if i < len(t_steps) - 1:
            x, _ = net.add_noise(x, t_steps[i + 1])

    return x
