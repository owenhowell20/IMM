import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--model", type=str, default="SongUNet", help="Base Model")

    parser.add_argument(
        "--dataset", type=str, default="ModelNet", help="Task to train on"
    )

    ### batch and group size
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_groups",
        type=int,
        default=8,
        help="Number of groups used in IMM sampling",
    )

    ### training parameters
    parser.add_argument(
        "--kernel", type=str, default="l2_kernel", help="Kernel type used in training"
    )

    # Meta-parameters
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--eta_min", type=float, default=0.1, help="Eta min")
    parser.add_argument(
        "--grad_clip", type=bool, default=True, help="use gradent clipping"
    )
    parser.add_argument(
        "--T_0", type=int, default=500, help="Restart Period for Optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="lr weight decay"
    )

    ### model args
    parser.add_argument(
        "--noise_schedule", type=str, default="fm", help="Noise Schedule"
    )

    # Logging
    parser.add_argument("--name", type=str, default="IMM training", help="Run name")
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

    ### restarts and model save directory
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models",
        help="Directory name to save models",
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
    parser.add_argument("--seed", type=int, default=1997)

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
