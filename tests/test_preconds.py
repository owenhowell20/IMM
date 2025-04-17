import pytest
import torch
from src.preconds import IMMPrecond
from data import mock_data_image


@pytest.fixture
def preconditioner():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IMMPrecond(
        img_resolution=16,  # Image resolution.
        img_channels=3,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        mixed_precision=None,
        noise_schedule="fm",
        model_type="SongUNet",
        sigma_data=0.5,
        f_type="euler_fm",
        T=0.994,
        eps=0.0,
        temb_type="identity",
        time_scale=1000.0,
    ).to(device)
    return model


def test_image_preconditioner_forward_step(preconditioner, mock_data_image):
    # Dummy input image
    x = mock_data_image
    device = "cpu"

    # Dummy timesteps
    batch_size = x.shape[0]
    s = torch.full((batch_size,), 0.8, device=device, dtype=torch.float32)
    t = torch.full((batch_size,), 0.6, device=device, dtype=torch.float32)

    ### TODO: add assert that s<t or t<s, i.e. dt > 0 always be positive

    ### get logsnr; should be batch size
    logsnr = preconditioner.get_logsnr(t)
    assert logsnr.shape[0] == batch_size, "Logsnr not right shape"

    ### Run forward pass on preconditioner
    output = preconditioner.forward(x, t, s)
    assert output.shape == x.shape, "prediction not same size as input"

    # Optional: check gradient flow
    output.mean().backward()
    assert x.grad is not None, "Gradients should be computed for input x"

    ### run cfg: this takes input x at time s and push forward to time t
    F = preconditioner.cfg_forward(
        x=x,
        t=t,
        s=s,  ### TODO: check issue when s=None, otherwise can just set s=0
    )

    assert F.shape == x.shape, "Dimension mismatch"
