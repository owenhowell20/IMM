import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch_cfm as cfm


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)]  # Scale to [-1, 1]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
