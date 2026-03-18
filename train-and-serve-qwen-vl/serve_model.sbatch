#!/bin/bash
#SBATCH --job-name=qwen25vl-serve
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=./logs/serve_%j.out
#SBATCH --error=./logs/serve_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
export MODEL_PATH="./models/qwen25vl-7b"

# ── Server config ─────────────────────────────────────────────────────────────
export HOST="0.0.0.0"
export PORT="8000"
export GPU_COUNT="1"          # increase and set --gres=gpu:N above to match
export MAX_MODEL_LEN="8192"

# ── NCCL (TCP-only, no InfiniBand on Crusoe) ──────────────────────────────────
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export NODE_HOSTNAME=$(hostname)

echo "======================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${NODE_HOSTNAME}"
echo "Model:      ${MODEL_PATH}"
echo "Host:       ${HOST}:${PORT}"
echo "GPUs:       ${GPU_COUNT}"
echo "MaxLen:     ${MAX_MODEL_LEN}"
echo "======================================"
echo ""
echo "API base URL for clients:  http://${NODE_HOSTNAME}:${PORT}/v1"
echo ""

mkdir -p ./logs

source .venv/bin/activate

python serve_model.py

echo "Job ${SLURM_JOB_ID} complete"
