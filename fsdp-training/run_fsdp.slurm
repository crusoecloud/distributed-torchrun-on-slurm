#!/bin/bash
#SBATCH --job-name=pytorch-fsdp-training
#SBATCH --nodes=6                    # Number of nodes (6 hosts)
#SBATCH --ntasks-per-node=1          # One task per node (torchrun handles the rest)
#SBATCH --gpus-per-node=8            # Number of GPUs per node (8x GPUs per node)
#SBATCH --cpus-per-task=64           # CPU cores per task
#SBATCH --time=02:00:00              # Time limit hrs:min:sec
#SBATCH --output=logs/slurm-%j.out   # Standard output log
#SBATCH --error=logs/slurm-%j.err    # Standard error log
#SBATCH --exclusive                  # Exclusive node access
#SBATCH --mem=0                      # Request all memory on each node

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 8))"

# Activate your Python environment
source ~/distributed-torchrun-on-slurm/.venv/bin/activate

# On ARM systems e.g GB200 use the system NCCL as installed by deb
# not the one that comes with PyTorch
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2           # Enable GPUDirect RDMA if available
#export NCCL_SOCKET_IFNAME=^docker,lo  # Exclude docker and loopback interfaces
export NCCL_IB_HCA=^mlx5_0:1

# Get the master node hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Training script arguments for FSDP training
EPOCHS=5000                    # More epochs to ensure lengthy training
BATCH_SIZE=300                 # Large batch size to utilize GPU memory
LEARNING_RATE=0.001
IMAGE_SIZE=288                 # High resolution images for Vision Transformer
NUM_SAMPLES=10000              # Synthetic dataset size per epoch
DATA_DIR="./data"
CHECKPOINT_DIR="./checkpoints_fsdp"
MODEL_NAME="vit_large"         # Large Vision Transformer model
SHARDING_STRATEGY="full_shard" # FSDP sharding strategy: full_shard, shard_grad_op, no_shard, hybrid_shard

# Run torchrun on each node with FSDP
# FSDP automatically shards model parameters, gradients, and optimizer states
# This will launch 6 nodes x 8 GPUs = 48 GPUs total
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_fsdp.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --data-dir $DATA_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --image-size $IMAGE_SIZE \
    --num-samples $NUM_SAMPLES \
    --model $MODEL_NAME \
    --sharding-strategy $SHARDING_STRATEGY

echo "Job finished at: $(date)"
