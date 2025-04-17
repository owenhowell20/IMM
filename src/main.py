# Get the parent directory
import os
import sys
from torch_geometric.datasets import ModelNet
from src.preconds import IMMPrecond
from src.loss import IMM_loss

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, parent_dir)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import wandb

from torch import optim
from torch.utils.data import DataLoader, random_split

from src.IMM_flags import get_flags


def to_np(x):
    return x.cpu().detach().numpy()


def train_epoch(epoch, model, dataloader, optimizer, schedule, FLAGS):
    model = model.to(FLAGS.device)
    model.train()

    num_iters = len(dataloader)
    assert num_iters > 0, "data not loaded correctly"
    loss_log = []
    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=True)

    for i, g in enumerate(dataloader):
        labels = g["labels"].to(FLAGS.device)
        data = g["data"].to(FLAGS.device)

        ### evaluate IMM loss
        optimizer.zero_grad()
        loss = IMM_loss(data, labels, model, FLAGS)
        loss.backward()

        ### log grad norms
        if i % FLAGS.print_interval == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    wandb.log({f"gradient_norm/{name}": grad_norm})

        optimizer.step()

        loss_log.append(to_np(loss))
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        if i % FLAGS.log_interval == 0:
            wandb.log({"Train Batch Loss": loss.item()}, commit=True)

        schedule.step(epoch + i / num_iters)

    # Loss logging
    average_loss = np.mean(np.array(loss_log))
    wandb.log({"Train Epoch Loss": average_loss.item()}, commit=False)


# ### evaluate on test set
# def evaluate_model(model, dataloader, FLAGS):
#     model = model.to(FLAGS.device)
#     if hasattr(model, "eval"):
#         model.eval()
#
#     num_iter = len(dataloader)
#
#     losses = []
#     for i, g in enumerate(dataloader):
#
#         key = g["key"].to(FLAGS.device)
#         value = g["value"].to(FLAGS.device)
#         question = g["question"].to(FLAGS.device)  ## (b, 3)
#         ### form model input:
#         result = torch.cat((key, value, question.unsqueeze(1)), dim=1)
#         answer = g["answer"].to(FLAGS.device)  ### (b,3)
#
#         pred = model(result)
#         loss = loss_function(pred, answer)
#         losses.append(to_np(loss))
#
#         wandb.log({"Test Batch Loss": loss.item()}, commit=True)
#
#     average_loss = np.mean(np.array(losses))
#     wandb.log({"Test Epoch Loss": average_loss.item()}, commit=True)
#
#     return True


def main(FLAGS, UNPARSED_ARGV):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"

    ### Instantiate the vector dataset
    dataset_path = FLAGS.dataset_path
    vector_dataset = ModelNet(root=dataset_path, name="40", train=True, transform=None)

    ### Define the split sizes
    train_ratio = 0.8
    total_size = len(vector_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    ### Split the dataset
    train_dataset, test_dataset = random_split(vector_dataset, [train_size, test_size])

    ### Wrap the datasets in DataLoaders
    dataloader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=FLAGS.num_workers,
        drop_last=False,
    )

    ### IMM preconditioner
    model = IMMPrecond(
        img_resolution=FLAGS.img_resolution,  # Image resolution.
        img_channels=FLAGS.img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        mixed_precision=None,
        noise_schedule=FLAGS.noise_schedule,
        model_type=FLAGS.model_type,
        sigma_data=0.5,
        f_type="euler_fm",
        T=0.994,
        eps=0.0,
        temb_type="identity",
        time_scale=1000.0,
    ).to(device)

    if isinstance(model, torch.nn.Module):
        model.to(device)

    ### check all layers trainable
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is non-trainable.")
        else:
            print(f"Layer {name} is trainable.")

    wandb.watch(model, log="all", log_freq=FLAGS.log_interval)

    optimizer = optim.AdamW(
        model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=FLAGS.T_0, eta_min=0.1 * FLAGS.lr
    )
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")

    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, dataloader, optimizer, scheduler, FLAGS)

        ### TODO: need to add the model eval
        # evaluate_model(
        #     model, dataloader=test_dataloader, FLAGS=FLAGS
        # )


### TODO: logging error, need to make sure record global step instead!!!
if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for run in range(FLAGS.num_runs):
        wandb.init(
            project="Equi-IMM",
            name=f"{FLAGS.name}",
            config=vars(FLAGS),
            reinit=True,
        )
        main(FLAGS, UNPARSED_ARGV)
        FLAGS.seed += 1
