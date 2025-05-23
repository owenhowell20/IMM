import torch
from src.kernels import l2_kernel


def IMM_loss(model, data, FLAGS, labels=None):
    """
    Computes the IMM loss.

    Args:
        model: The model used for loss computation.
        data (torch.Tensor): Input data of shape [M*b, ...] where M is the number of samples per group.
        FLAGS: Configuration containing hyperparameters, including M.
        labels (torch.Tensor, optional): Labels for conditional training, shape [M*b, ...].

    Returns:
        torch.Tensor: total IMM loss across all groups.
    """
    M = FLAGS.M  # Number of samples per group
    total_batch = data.shape[0]
    batch_size = total_batch // M  # Ensure batch_size is an integer

    # Validate batch_size
    assert (
        total_batch % M == 0
    ), f"Total batch size {total_batch} must be divisible by M={M}."

    # Initialize loss
    total_loss = 0

    # Loop over the number of groups
    for i in range(batch_size):
        group_data = data[i * M : (i + 1) * M]

        # Ensure each group has exactly M samples
        assert group_data.shape[0] == M, f"Each group must have exactly M={M} samples!"

        # Compute the IMM loss for this group
        total_loss += IMM_loss_per_group(group_data, labels, model, FLAGS)

    return total_loss


def IMM_loss_per_group(model, data, labels=None):
    """
    Computes IMM loss for a single group of data.

    Args:
        model: The model used for the loss computation.
        data (torch.Tensor): Input data of shape [M, ..., ...].
        labels (torch.Tensor, optional): Labels for conditional training.

    Returns:
        torch.Tensor: The IMM loss for this group, weighted by the factor `weight_s_t`.
    """
    batch_size = data.shape[0]

    # Sample from the prior (noise)
    noise = torch.randn_like(data)

    one = torch.ones(batch_size, device=data.device, dtype=torch.float32)
    t = torch.rand(
        batch_size, device=data.device, dtype=torch.float32
    )  # Random t in [0.0, 1.0]

    # DDIM: x_{t} <- DDIM(noise, x, t, 1)
    x_t = model.ddim(noise, data, t, one)

    assert (
        x_t.shape == data.shape
    ), f"Expected x_t shape {data.shape}, but got {x_t.shape}"

    # DDIM: x_{r} <- DDIM(x_{t}, x, r, t)
    r = (
        torch.rand(batch_size, device=data.device, dtype=torch.float32) * t
    )  # Random r in [0.0, t_i]
    x_r = model.ddim(x_t, data, r, t)

    # Compute the loss (Maximum Mean Discrepancy)
    loss = MMD_loss(model, x_r, x_t, r, t, labels=labels)

    # Weight for the IMM loss
    weight_s_t = 1  # TODO: not sure what correct choice of weights is

    return weight_s_t * loss


def MMD_loss(model, x_r, x_t, s, r, t, labels=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between two sets of forward function outputs:
    one computed with gradients (`f_{s,t}(x_t)`) and one without gradients (`f_{s,r}(x_r)`).

    Args:
        model: The model used for the forward computation.
        x_r (torch.Tensor): The data generated by DDIM at `r` (shape [M, ...]).
        x_t (torch.Tensor): The data generated by DDIM at `t` (shape [M, ...]).
        s (torch.Tensor): The scaling factor for both forward passes.
        r (torch.Tensor): Time tensor used in the no-grad forward pass.
        t (torch.Tensor): Time tensor used in the grad-forward pass.
        labels (torch.Tensor, optional): Labels for conditional training.

    Returns:
        torch.Tensor: The computed MMD loss between the two forward pass outputs.
    """
    # Forward pass with gradient: f_{s,t}(x_t)
    f_grad = model.cfg_forward(x=x_t, t=t, s=s)

    # Forward pass without gradient: f_{s,r}(x_r)
    with torch.no_grad():
        f_no_grad = model.cfg_forward(x=x_r, t=r, s=s)

    # Ensure the outputs have the same shape
    assert (
        f_grad.shape == f_no_grad.shape
    ), "Grad and No Grad terms have different shapes."

    # Compute and return the L2 norm of the difference between the outputs
    return torch.norm(f_grad - f_no_grad, p=2, dim=tuple(range(1, f_grad.ndim)))
