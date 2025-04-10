import pytest
import torch
from src.preconds import IMMPrecond  # adjust import as needed
from data import mock_data


@pytest.fixture
def preconditioner():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IMMPrecond(
        img_resolution=256,
        img_channels=3,
        label_dim=0,
        mixed_precision=None,
        noise_schedule="fm",
        model_type="SongUNet",  # make sure this class exists
        sigma_data=0.5,
        f_type="euler_fm",
        T=0.994,
        eps=0.0,
        time_scale=1000.0,
        temb_type="identity",
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=[1],
        num_res_blocks=1,
        use_spatial_transformer=False,
    ).to(device)
    return model


def test_preconditioner_forward_step(preconditioner, mock_data):
    # Dummy input image
    x = mock_data

    # Dummy timesteps
    t = torch.full((batch_size,), 0.8, device=device, dtype=torch.float32)
    s = torch.full((batch_size,), 0.6, device=device, dtype=torch.float32)

    # Run forward pass
    output = preconditioner.forward(x, t, s)

    # Basic checks
    assert output.shape == x.shape, "Output shape must match input shape"
    assert output.dtype == torch.float32, "Output dtype should match input"
    assert not torch.isnan(output).any(), "Output contains NaNs"

    # Optional: check gradient flow
    output.mean().backward()
    assert x.grad is not None, "Gradients should be computed for input x"
