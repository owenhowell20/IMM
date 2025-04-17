import pytest
import torch


@pytest.fixture
def mock_data_image():
    batch_size = 2
    channels = 3
    height = width = 16
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


@pytest.fixture
def mock_data_pc():
    batch_size = 2
    num_points = 8
    dimension = 4  ### point cloud feature dimension
    data = torch.randn(
        batch_size,
        num_points,
        3 + dimension,  ### (coords + node features)
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    return data
