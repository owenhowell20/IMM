import pytest
import torch
from src.preconds import IMMPrecond
from data import mock_data_image


from src.loss import MMD_loss, IMM_loss_per_group, IMM_loss


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


# def test_MMD_loss_per_group(preconditioner, mock_data_image):
#
#     loss = IMM_loss_per_group(preconditioner, mock_data_image)
#
#     print(loss.shape)
#     assert False
#


def test_MMD_loss(preconditioner, mock_data_image):
    device = mock_data_image.device
    batch_size = mock_data_image.shape[0]

    x_r = torch.randn_like(mock_data_image)
    x_t = torch.randn_like(x_r)

    s = torch.full((batch_size,), 0.1, device=device, dtype=torch.float32)
    r = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
    t = torch.full((batch_size,), 0.8, device=device, dtype=torch.float32)

    loss = MMD_loss(preconditioner, x_r, x_t, s, r, t)
    assert loss.shape[0] == batch_size, "MMD loss wrong shape"
