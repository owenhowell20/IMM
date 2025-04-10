import pytest
import torch


@pytest.fixture
def mock_data():
    batch_size = 2
    channels = 3
    height = width = 256
    data = torch.randn(
        batch_size,
        channels,
        height,
        width,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    return data
