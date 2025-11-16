#!/usr/bin/env python3
"""
Distributed PyTorch Training Example for Multi-Node, Multi-GPU Setup
Uses torchrun for launching distributed training across SLURM nodes
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import argparse
import time


def setup_distributed():
    """Initialize the distributed environment."""
    # torchrun sets these environment variables
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Initialize process group
    dist.init_process_group(backend='nccl')

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


class SimpleConvNet(nn.Module):
    """Simple CNN for demonstration purposes."""
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


def get_dataloader(rank, world_size, batch_size, data_dir='./data'):
    """Create distributed dataloaders."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Wait for rank 0 to finish downloading
    if world_size > 1:
        dist.barrier()

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, train_sampler


def train_epoch(model, dataloader, optimizer, criterion, epoch, rank, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(local_rank), target.cuda(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += len(data)

        if batch_idx % 100 == 0 and rank == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] '
                  f'Loss: {loss.item():.6f}')

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * total_correct / total_samples

    # Aggregate metrics across all processes
    metrics = torch.tensor([avg_loss, accuracy], device=local_rank)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= dist.get_world_size()

    if rank == 0:
        print(f'Epoch {epoch} - Avg Loss: {metrics[0]:.6f}, Accuracy: {metrics[1]:.2f}%')

    return metrics[0].item(), metrics[1].item()


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, rank):
    """Save checkpoint (only on rank 0)."""
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Save the underlying model
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')


def main():
    parser = argparse.ArgumentParser(description='Distributed PyTorch Training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(f'Starting distributed training on {world_size} GPUs')
        print(f'Arguments: {args}')

    # Create model and move to GPU
    model = SimpleConvNet(num_classes=10).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.NLLLoss()

    # Create dataloader
    train_loader, train_sampler = get_dataloader(
        rank, world_size, args.batch_size, args.data_dir
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        start_time = time.time()
        avg_loss, accuracy = train_epoch(
            model, train_loader, optimizer, criterion, epoch, rank, local_rank
        )
        epoch_time = time.time() - start_time

        if rank == 0:
            print(f'Epoch {epoch} completed in {epoch_time:.2f}s')

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, rank)

        # Synchronize all processes
        dist.barrier()

    if rank == 0:
        print('Training completed!')

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
