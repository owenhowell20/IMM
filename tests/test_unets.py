from src.unets import SongUNet, DhariwalUNet
from data import mock_data_image


def test_SongUNet(mock_data_image):
    img_resolution = 16
    in_channels = 3
    out_channels = 3

    model = SongUNet(
        img_resolution=img_resolution,  # Image resolution at input/output.
        in_channels=in_channels,  # Number of color channels at input.
        out_channels=out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        s_embed=True,
        share_tsemb=True,
        embedding_kwargs={},
    )

    x = mock_data_image
    noise_labels_t = (None,)
    noise_labels_s = (None,)
    class_labels = (None,)
    augment_labels = None

    output = model.forward(
        x=x,
        noise_labels_t=noise_labels_t,
        noise_labels_s=noise_labels_s,
        class_labels=class_labels,
        augment_labels=augment_labels,
    )

    assert output.shape == x.shape, "Output Size is incorrect"
