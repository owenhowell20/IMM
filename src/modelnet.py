import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.loader import DataLoader

# Parameters
num_points = 1024  # Number of points per point cloud
batch_size = 32
dataset_path = "./data/ModelNet40"  # You can change this path

# Transform: sample N points per shape
transform = SamplePoints(num_points)

# Load datasets
train_dataset = ModelNet(root=dataset_path, name="40", train=True, transform=transform)
test_dataset = ModelNet(root=dataset_path, name="40", train=False, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Check sample
if __name__ == "__main__":
    for batch in train_loader:
        print(batch)
        print(f"Batch pos shape: {batch.pos.shape}")  # [batch_size * num_points, 3]
        print(f"Batch y shape: {batch.y.shape}")  # [batch_size]
        break
