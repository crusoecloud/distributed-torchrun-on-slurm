# PyTorch Distributed Training on SLURM

This example demonstrates how to run distributed PyTorch training across multiple SLURM nodes, each with multiple GPUs, using `torchrun`.

## Overview

- **Setup**: 4 nodes, 8 GPUs per node = 16 total GPUs - edit to match the capacity of your cluster
- **Launcher**: `torchrun` (PyTorch's recommended distributed launcher)
- **Backend**: NCCL for GPU communication
- **Dataset**: MNIST (for demonstration)



## Requirements

Install the required packages in a virtual environment accessible to all hosts
(if using the Slurm solution from Crusoe github, or anything where all the slurm nodes share a common /home dir mounted from NFS,
you can create and activate the virtual env from the login node and install the requirements on that; then the sbatch file should also contain a step to
activate the venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 #or whichever version of cuda is suitable for the nvidia driver on your ci
pip install -r requirements.txt
```

## Configuration

### SLURM Script Configuration

cd into the directory for your chosen example e.g basic distributed training or the vision model
Edit `run_distributed.slurm` to match your cluster setup:


1. **Environment activation** (lines 24-27):
   ```bash
   source ~/distributed-torchrun-on-slurm/venv/bin/activate

   ```

2. **SLURM parameters** (lines 2-10):
   - `--nodes=4` - Number of nodes (adjust as needed)
   - `--gpus-per-node=8` - GPUs per node (adjust as needed)
   - `--cpus-per-task=32` - CPU cores (adjust based on your system)
   - `--time=02:00:00` - Time limit

3. **Network interface** (line 31):
   ```bash
   export NCCL_SOCKET_IFNAME=^docker,lo  # Adjust for your network setup
   ```
   Common interfaces: `eth0`, `ib0` (InfiniBand), `enp0s31f6`, etc.

## Usage

### 1. Submit the SLURM Job

```bash
# Make sure the logs directory exists
mkdir -p logs

# Submit the job
sbatch run_distributed.slurm
```

### 2. Monitor the Job

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/slurm-<job_id>.out

# View live errors
tail -f logs/slurm-<job_id>.err
```

### 3. Cancel the Job (if needed)

```bash
scancel <job_id>
```

## How It Works

### Distributed Setup

The training script uses PyTorch's Distributed Data Parallel (DDP):

1. **torchrun** launches one process per GPU (16 total processes)
2. Each process gets environment variables:
   - `RANK` - Global rank (0-15)
   - `LOCAL_RANK` - Local rank on the node (0-7)
   - `WORLD_SIZE` - Total number of processes (16)

3. **NCCL** handles inter-GPU communication for:
   - Gradient synchronization
   - Metric aggregation
   - Barrier synchronization

### Data Distribution

- **DistributedSampler** splits the dataset across all GPUs
- Each GPU processes a unique subset of the data
- Gradients are averaged across all GPUs after each backward pass

### Key Components

From `train_distributed.py`:

- `setup_distributed()` - Initializes the process group
- `DDP(model)` - Wraps the model for distributed training
- `DistributedSampler` - Distributes data across GPUs
- `dist.all_reduce()` - Aggregates metrics across all processes
- `dist.barrier()` - Synchronizes all processes

## Customization

### Training Parameters

Edit in `run_distributed.slurm` (lines 42-46):

```bash
EPOCHS=10
BATCH_SIZE=64           # Per-GPU batch size
LEARNING_RATE=0.01
DATA_DIR="./data"
CHECKPOINT_DIR="./checkpoints"
```

### Model Architecture

Edit the `SimpleConvNet` class in `train_distributed.py` to use your own model.

### Dataset

Replace the MNIST dataset in `get_dataloader()` with your own dataset:

```python
train_dataset = datasets.ImageFolder(
    root=args.data_dir,
    transform=transform
)
```

## Troubleshooting

### Common Issues

1. **NCCL Timeout or Initialization Failure**
   - Check network connectivity between nodes
   - Verify firewall allows communication on MASTER_PORT
   - Set `export NCCL_DEBUG=INFO` for detailed logs

2. **Out of Memory (OOM)**
   - Reduce `--batch-size`
   - Reduce model size
   - Enable gradient checkpointing

3. **Slow Training**
   - Check if InfiniBand is enabled (`NCCL_IB_DISABLE=0`)
   - Verify GPUDirect RDMA is available (`NCCL_NET_GDR_LEVEL=2`)
   - Increase `num_workers` in DataLoader

4. **Hanging at Initialization**
   - Ensure all nodes can reach the master node
   - Check `NCCL_SOCKET_IFNAME` is set correctly
   - Verify SLURM environment variables are set

### Debugging

Enable verbose logging:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## Performance Tips

1. **Optimize batch size**: Use the largest batch size that fits in GPU memory
2. **Pin memory**: Already enabled in DataLoader (`pin_memory=True`)
3. **Multiple workers**: Adjust `num_workers` in DataLoader based on CPU count
4. **Mixed precision**: Add automatic mixed precision (AMP) for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Scaling to More Nodes/GPUs

To scale to different configurations, only change the SLURM parameters:

```bash
#SBATCH --nodes=4                    # 4 nodes
#SBATCH --gpus-per-node=8            # 8 GPUs per node = 32 total GPUs
```

The training script automatically adapts to the number of available GPUs.

## Checkpoints

Checkpoints are saved after each epoch to `./checkpoints/`:

```bash
checkpoints/
  checkpoint_epoch_1.pt
  checkpoint_epoch_2.pt
  ...
```

Only rank 0 saves checkpoints to avoid conflicts.

## Verifying Distributed Training

Check the output logs for:

```
Starting distributed training on 16 GPUs
Master node: node001
...
Epoch 1 - Avg Loss: 0.234567, Accuracy: 92.34%
```

The number of GPUs should match: `nodes * gpus-per-node = total GPUs`

## Additional Resources

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
