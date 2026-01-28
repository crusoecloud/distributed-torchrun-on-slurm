Running TorchTitan on Crusoe Cloud's SLURM Solution

From the TorchTitan repo (https://github.com/pytorch/torchtitan):

*Torchtitan is a PyTorch native platform designed for rapid experimentation and large-scale training of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, torchtitan provides a flexible foundation for developers to build upon. With torchtitan extension points, one can easily create custom extensions tailored to specific needs.*

We will do multinode fine tuning of Llama3_8b using the C4 dataset. These instructions are specific to GB200, but the general principle is the same for any Crusoe Slurm cluster. The results section at the bottom of the page will be extended to cover multiple GPUs and cluster and batch sizes.

Edit torchtitan/models/llama3/train_configs/llama3_8b.toml to enable checkpointing (if required - to me, seeing checkpoints created is a tangible measure of success!)
Download the tokenizer from huggingface as shown in the TorchTitan instructions
On any one of the compute nodes (not the login or head node!) create a Python virtual environment, activate it, and install the requirements:


sudo apt-get install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Update multinode_trainer.slurm for the correct number of nodes and gpus per node (in both the ‘sbatch’ parts and in the torchrun command near the bottom) and add a line to activate the venv:

```                                                                                                                                                                                                                  
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
```
Run the job from the slurm login node: sbatch multinode_trainer.sbatch. ‘Watch squeue’ to see that the job is running - if it doesn’t go to ‘R’ status it could be that you didn’t have sufficient nodes in ‘idle’ state, or that you request resources that no node has (e.g too many GPU or CPU per node)
When the job is running, its logs will be written to slurm-x.out where x is the ID of the slurm job. The outputs below show the performance to be expected on a cluster where all the nodes are in the same imex partition.

### Example config file for 90% memory usage on GB200 ###

```
[job]
dump_folder = "./outputs"
description = "Llama 3 8B training"

[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"
profile_freq = 100

[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "8B"
hf_assets_path = "./assets/hf/Llama-3.1-8B"
# converters = ["float8"]

[optimizer]
name = "AdamW"
lr = 3e-4
eps = 1e-8

[lr_scheduler]
warmup_steps = 200  # lr scheduler warm up

[training]
local_batch_size = 5
seq_len = 8192
max_norm = 1.0  # grad norm clipping
steps = 100
dataset = "c4"
dataset_path = "/data/c4/"

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
context_parallel_degree = 1

[checkpoint]
enable = false
folder = "checkpoint"
interval = 500
last_save_model_only = true
export_dtype = "float32"
async_mode = "disabled" # ["disabled", "async", "async_with_pinned_mem"]

[compile]
enable=false
components = ["model", "loss"]

[activation_checkpoint]
mode = "selective"  # ["none", "selective", "full"]
selective_ac_option = "op"  # "int" = ac every positive int layer or 'op', ac based on ops policy

[quantize.linear.float8]
enable_fsdp_float8_all_gather = false
precompute_float8_dynamic_scale_for_fsdp = false
filter_fqns = ["output"]
```

# TorchTitan performance results for various cluster configs #

|GPU Type|Compute nodes in training job|Batch size|Performance indicators at step 1000 of Torchtitan test described on this page|
|--------|-----------------------------|----------|-----------------------------------------------------------------------------|
|GB200   |17                           |5         |step: 100  loss:  6.4277  grad_norm:  2.9062  memory: 158.61GiB(86.20%)  tps: 17,581  tflops: 1,018.20  mfu: 45.25%|
|GB200   |9                            |5         |step: 100  loss:  8.2954  grad_norm: 29.3788  memory: 159.31GiB(86.58%)  tps: 17,975  tflops: 1,040.99  mfu: 46.27%|
|GB200   |3                            |5         |step: 100  loss:  6.5572  grad_norm:  3.6680  memory: 161.65GiB(87.85%)  tps: 18,284  tflops: 1,058.93  mfu: 47.06%|

