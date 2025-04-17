import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--model", type=str, default="Hyena", help="String name of model"
    )

    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--sequence_length", type=int, default=4, help="Sequence Size")
    parser.add_argument(
        "--num_sequences", type=int, default=8000, help="Number of Seqences"
    )

    # Meta-parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument(
        "--grad_clip", type=bool, default=False, help="use gradent clipping"
    )
    parser.add_argument(
        "--use_mask_tokens", type=bool, default=False, help="Random Mask on tokens"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="lr weight decay"
    )

    # location of data for relational inference
    ## TODO: dont need these?
    parser.add_argument("--ri_data", type=str, default="data_generation")
    parser.add_argument("--data_str", type=str, default="my_datasetfile")

    ### model args
    parser.add_argument(
        "--positional_encoding_dimension",
        type=int,
        default="64",
        help="dimension of positional encodings",
    )
    parser.add_argument("--input_dimension_1", type=int, default="128")
    parser.add_argument("--input_dimension_2", type=int, default="64")
    parser.add_argument("--input_dimension_3", type=int, default="32")

    # Logging
    parser.add_argument("--name", type=str, default="big model test", help="Run name")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Number of steps between logging key stats",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=250,
        help="Number of steps between printing key stats",
    )

    ### restart and model save
    parser.add_argument(
        "--save_dir", type=str, default="models", help="Directory name to save models"
    )
    parser.add_argument(
        "--restore", type=str, default=None, help="Path to model to restore"
    )
    parser.add_argument("--verbose", type=int, default=0)

    # Miscellanea
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Exit after 10 steps for profiling"
    )

    # Random seed for both Numpy and Pytorch
    parser.add_argument("--seed", type=int, default=1992)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    if torch.cuda.is_available():
        FLAGS.device = torch.device("cuda:0")
    else:
        FLAGS.device = torch.device("cpu")

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    return FLAGS, UNPARSED_ARGV
