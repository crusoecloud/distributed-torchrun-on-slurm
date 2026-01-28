#!/bin/bash
#SBATCH --job-name=torchtitan_multi_node
#SBATCH --ntasks=18
#SBATCH --nodes=18
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=96
#SBATCH --partition=batch

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

source ~/torchtitan/.venv/bin/activate

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 18 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
dcgmi profile --resume
