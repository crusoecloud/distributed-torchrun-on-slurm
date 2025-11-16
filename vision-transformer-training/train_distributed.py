#!/usr/bin/env python3
"""
Distributed PyTorch Training for Multi-Node, Multi-GPU Setup
Uses Vision Transformer with high-resolution synthetic images for lengthy H200 GPU training
Optimized for 6 nodes x 8 H200 GPUs = 48 GPUs total
"""

import os
import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.cuda.amp import autocast, GradScaler
import argparse
import time
import math
import numpy as np


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


class SyntheticImageDataset(Dataset):
    """
    Generates synthetic high-resolution images on-the-fly.
    This ensures GPU compute is the bottleneck, not data loading.
    """
    def __init__(self, num_samples, image_size=384, num_classes=1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        # Use a fixed seed per sample for reproducibility but variety
        self.seeds = np.arange(num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image with complex patterns
        # This simulates realistic image data with spatial structure
        np.random.seed(self.seeds[idx])

        # Create complex synthetic image with multiple frequency components
        image = np.random.randn(3, self.image_size, self.image_size).astype(np.float32)

        # Add some structure to make it more realistic
        for _ in range(3):
            image = image + 0.3 * np.roll(image, shift=1, axis=1)
            image = image + 0.3 * np.roll(image, shift=1, axis=2)

        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Generate label
        label = idx % self.num_classes

        return torch.from_numpy(image), label


class VisionTransformer(nn.Module):
    """
    Large Vision Transformer model optimized for H200 GPUs.
    This model is computationally intensive to ensure full GPU utilization.
    """
    def __init__(
        self,
        image_size=384,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)  # (B, dim, H/P, W/P)
        x = x.transpose(1, 2)  # (B, num_patches, dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Classification
        x = self.norm(x[:, 0])  # Use cls token
        x = self.fc(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def create_model(model_name, image_size, num_classes=1000):
    """Create model based on model name."""
    if model_name == "vit_large":
        model = VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096,
            dropout=0.1
        )
    elif model_name == "vit_huge":
        model = VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=1280,
            depth=32,
            heads=16,
            mlp_dim=5120,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_dataloader(rank, world_size, batch_size, num_samples, image_size, data_dir='./data'):
    """Create distributed dataloaders with synthetic high-resolution images."""
    # Create synthetic dataset
    train_dataset = SyntheticImageDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_classes=1000
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # Create dataloader with optimized settings for H200
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,  # Increased for H200
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, train_sampler


def train_epoch(model, dataloader, optimizer, criterion, scaler, epoch, rank, local_rank,
                gradient_accumulation_steps=1):
    """
    Train for one epoch with mixed precision and gradient accumulation.
    Optimized for H200 GPU utilization.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()

    start_time = time.time()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/my_run'),
        record_shapes=True,
        with_stack=True
    ) as prof:

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(local_rank, non_blocking=True), target.cuda(local_rank, non_blocking=True)

            # Mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output, target)
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Calculate accuracy
            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct = pred.eq(target).sum().item()

            total_loss += loss.item() * gradient_accumulation_steps
            total_correct += correct
            total_samples += len(data)

            # Print progress
            if batch_idx % 10 == 0 and rank == 0:
                samples_per_sec = total_samples / (time.time() - start_time)
                print(f'Epoch {epoch} [{batch_idx}/{len(dataloader)}] '
                      f'Loss: {loss.item() * gradient_accumulation_steps:.6f} '
                      f'Speed: {samples_per_sec:.2f} samples/sec')

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * total_correct / total_samples

    # Aggregate metrics across all processes
    metrics = torch.tensor([avg_loss, accuracy, total_samples], device=local_rank)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    avg_loss_global = metrics[0].item() / dist.get_world_size()
    accuracy_global = metrics[1].item() / dist.get_world_size()
    total_samples_global = int(metrics[2].item())

    epoch_time = time.time() - start_time

    if rank == 0:
        throughput = total_samples_global / epoch_time
        print(f'Epoch {epoch} Summary:')
        print(f'  Avg Loss: {avg_loss_global:.6f}')
        print(f'  Accuracy: {accuracy_global:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  Throughput: {throughput:.2f} samples/sec ({throughput * dist.get_world_size():.2f} global)')
        print(f'  Total samples: {total_samples_global}')

    return avg_loss_global, accuracy_global


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
    parser = argparse.ArgumentParser(
        description='Distributed PyTorch Training with Vision Transformer for H200 GPUs'
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--image-size', type=int, default=128, help='Image size (height and width)')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of synthetic samples per epoch')
    parser.add_argument('--model', type=str, default='vit_large', choices=['vit_large', 'vit_huge'],
                        help='Model architecture')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N epochs')
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print('='*80)
        print(f'Distributed Training on H200 GPUs')
        print('='*80)
        print(f'World size: {world_size} GPUs')
        print(f'Nodes: {world_size // 8}')
        print(f'GPUs per node: 8')
        print(f'Model: {args.model}')
        print(f'Image size: {args.image_size}x{args.image_size}')
        print(f'Batch size per GPU: {args.batch_size}')
        print(f'Global batch size: {args.batch_size * world_size}')
        print(f'Epochs: {args.epochs}')
        print(f'Samples per epoch: {args.num_samples}')
        print('='*80)

    # Create model and move to GPU
    if rank == 0:
        print(f'Creating {args.model} model...')

    model = create_model(args.model, args.image_size, num_classes=1000)
    model = model.cuda(local_rank)

    # Calculate model parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        print(f'Model parameters: {num_params:,}')
        print(f'Trainable parameters: {num_trainable_params:,}')
        print(f'Model size: ~{num_params * 4 / 1024**3:.2f} GB (fp32)')

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler
    scaler = GradScaler()

    # Create dataloader
    if rank == 0:
        print('Creating dataloaders...')

    train_loader, train_sampler = get_dataloader(
        rank, world_size, args.batch_size, args.num_samples, args.image_size, args.data_dir
    )

    if rank == 0:
        print(f'Batches per epoch: {len(train_loader)}')
        print('='*80)
        print('Starting training...')
        print('='*80)

    # Training loop
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\nEpoch {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.6f})')

        # Train for one epoch
        avg_loss, accuracy = train_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch, rank, local_rank,
            args.gradient_accumulation_steps
        )

        # Update learning rate
        scheduler.step()

        # Save checkpoint periodically
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, rank)

        # Synchronize all processes
        dist.barrier()

        # Estimate time remaining
        if rank == 0:
            elapsed = time.time() - total_start_time
            avg_epoch_time = elapsed / epoch
            remaining_epochs = args.epochs - epoch
            eta = remaining_epochs * avg_epoch_time
            print(f'Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m')

    # Save final checkpoint
    save_checkpoint(model, optimizer, args.epochs, args.checkpoint_dir, rank)

    total_time = time.time() - total_start_time

    if rank == 0:
        print('='*80)
        print('Training completed!')
        print(f'Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
        print('='*80)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
