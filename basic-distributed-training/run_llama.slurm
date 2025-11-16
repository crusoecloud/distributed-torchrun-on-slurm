#!/bin/bash
#SBATCH --job-name=pytorch-llama-training
#SBATCH --nodes=6                    # Number of nodes
#SBATCH --ntasks-per-node=8          # One task per node (torchrun handles the rest)
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8            # Number of GPUs per node
#SBATCH --cpus-per-task=32           # CPU cores per task (adjust based on your system)
#SBATCH --time=02:00:00              # Time limit hrs:min:sec
#SBATCH --output=logs/slurm-%j.out   # Standard output log
#SBATCH --error=logs/slurm-%j.err    # Standard error log
#SBATCH --exclusive                  # Exclusive node access

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 8))"

# Activate your Python environment
source ~/.venv/bin/activate

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2           # Enable GPUDirect RDMA if available
# export NCCL_SOCKET_IFNAME=^docker,lo  # Exclude docker and loopback interfaces
export NCCL_IB_HCA=^mlx5_0:1


# Get the master node hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Training script arguments
EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=0.01
DATA_DIR="./data"
CHECKPOINT_DIR="./checkpoints"

# Run torchrun on each node
# torchrun automatically handles distributed setup
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_llama.py \

echo "Job finished at: $(date)"
