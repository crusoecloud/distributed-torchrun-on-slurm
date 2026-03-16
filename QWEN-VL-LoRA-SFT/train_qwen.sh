#!/bin/bash
#SBATCH --job-name=qwen25vl-sft-ds
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=./logs/sft_%j.out
#SBATCH --error=./logs/sft_%j.err

JOBID=$SLURM_JOB_ID
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

# ── NCCL (TCP-only, no InfiniBand on Crusoe) ──────────────────────────────────
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# ── Paths ─────────────────────────────────────────────────────────────────────
export MODEL_PATH="./models/qwen25vl-7b"
export OUTPUT_DIR="./outputs/qwen25vl-sft-${JOBID}"
export DATA_PATH="./data/llava_webdataset/pretrain*.tar"

# ── Training hyperparams ──────────────────────────────────────────────────────
# Effective batch = MICRO_BATCH * GRAD_ACCUM * NUM_GPUS
# = 4 * 4 * 16 = 256
export MICRO_BATCH=4
export GRAD_ACCUM=4
export MAX_STEPS=2000
export MAX_PIXELS=1048576   # 1024x1024 — uses ~70% of 80GB H100
export MAX_LENGTH=4096

# Enable NCCL flight recorder — gives stack traces on timeout instead of silence
export TORCH_FR_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# More verbose NCCL output
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# Catch distributed hangs with a stack trace
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

echo "======================================"
echo "Job:          ${JOBID}"
echo "Nodes:        ${SLURM_JOB_NUM_NODES}"
echo "Master:       ${MASTER_ADDR}"
echo "Model:        ${MODEL_PATH}"
echo "Shards:       ${WEBDATASET_SHARDS}"
echo "Output:       ${OUTPUT_DIR}"
echo "Steps:        ${MAX_STEPS}"
echo "MBS:          ${MICRO_BATCH}"
echo "GradAccum:    ${GRAD_ACCUM}"
echo "Eff. Batch:   $((MICRO_BATCH * GRAD_ACCUM * 16))"
echo "MaxPixels:    ${MAX_PIXELS}"
echo "MaxLength:    ${MAX_LENGTH}"
echo "======================================"

mkdir -p "$OUTPUT_DIR" ./logs

source .venv/bin/activate

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ./train_qwen.py

echo "Job ${JOBID} complete"
