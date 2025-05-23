import torch
from torch_utils import persistence


### Global point cloud encoder
@persistence.persistent_class
class pc_model(torch.nn.Module):
    def __init__(
        self,
        num_points,  # number of points
        node_features,  # node features dimension
        out_features=256,  # Number of color channels at output.
        embedding_type="fourier",
        encoder_type="standard",
        decoder_type="standard",
        **kwargs,
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()

    def forward(
        self,
        x,
        noise_labels_t,
        noise_labels_s=None,
        class_labels=None,
        augment_labels=None,
    ):
        return x
