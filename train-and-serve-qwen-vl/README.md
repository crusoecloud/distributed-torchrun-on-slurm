## Quick Start ##  

This example is tested for use on a Crusoe Managed Slurm cluster with a minimum of 1 x H100.8x GPU node, and relies on pre-existing Slurm and UV installations included in the Crusoe Managed Slurm product. It can be adapted for use with any other comparable Slurm cluster. [Install Crusoe Managed Slurm](https://docs.crusoecloud.com/orchestration/slurm/overview) if you haven't already done so.  

[Create users on your Crusoe Managed Slurm cluster](https://docs.crusoecloud.com/orchestration/slurm/user-management) and then SSH into the Login node as your Slurm user. Git clone this repo and cd into this directory.  

Make venv_setup.sh executable and run it, then download the model weights and the dataset (these steps take several minutes):
```
chmod a+x venv_setup.sh && ./venv_setup.sh
export HF_TOKEN=<your huggingface token>
python download_dataset.py
python download_model.py
```
Edit the sbatch file (train_qwen.sbatch) to have the correct number of worker nodes for your cluster.
Run the sbatch file:
```
sbatch train_qwen.sbatch
```
Tail the .err and .out files from the logs directory to monitor the progress of your job. You can use the Metrics tab of your GPU nodepool instances to monitor GPU metrics charts while the job is in progress. On a single node, the job as configured here (5000 steps) takes about 4.5 hours to complete and creates a checkpoint every 500 steps.  

Merge the resulting LoRA adapter into the base model to create a new version of the model:
```
source .venv/bin/activate #if not already done
OUTPUT_SUBDIR_NAME=<name of the dir under ./outputs with your training output> python merge_adapter.py
```
Now compare the base model with the fine-tuned model by serving them with vLLM and asking them to describe images. Provide the path of the model you want to serve and run the serve_model.sbatch script. Starting with the base model:
```
MODEL_PATH=./models/qwen25vl-7b serve_model.sbatch
```
Tail the corresponding logs file as vLLM starts up on the chosen worker node. When vLLM is ready to serve, you can run the curl examples while logged into the Slurm login node to see the results. Cancel the job and re-run it giving the path to the merged model's directory. You should see that your fine-tuned model describes the same image in different terms to the base model.

---
## Description ##  
The script fine-tunes Qwen2.5-VL-7B, a vision-language model, on a dataset of image-caption pairs stored as WebDataset tar archives.

  Data pipeline: At startup, each rank reads the RANK/WORLD_SIZE environment variables set by torchrun and takes every Nth shard from the sorted shard list (shard_files[rank::world_size]), so the full dataset is covered once across all GPUs with no overlap. Within each rank, shards are further divided
  across dataloader workers via split_by_worker. The pipeline flat-maps parse_webdataset over each tar sample, yielding one training example per image-caption pair. The dataset repeats indefinitely so training runs for a fixed number of steps rather than epochs. The QwenDataCollator applies the Qwen chat
  template to format each example as a user/assistant conversation, runs the processor to tokenise text and tile images into patches, and masks padding tokens in the labels so they don't contribute to loss.

  Training: The HuggingFace Trainer handles the DDP training loop across all 16 GPUs (2 nodes × 8). Key settings: dispatch_batches=False so each rank fetches its own batch independently (required for iterable datasets with variable-length sequences), ddp_broadcast_buffers=False to avoid spurious
  buffer-sync collectives, gradient checkpointing to trade recomputation for activation memory.

  ---
  Why DeepSpeed + Flash Attention + LoRA

  These three techniques attack the memory and compute bottlenecks from different angles:

  - LoRA dramatically reduces the number of trainable parameters. Instead of updating all 7B weights, low-rank adapter matrices (r=32) are inserted alongside the attention projections — only ~0.5% of parameters are trained. This cuts optimizer state memory (Adam stores two momentum tensors per trainable
  parameter) by roughly 200×, making it feasible to fit the model on H100s without running out of VRAM.
  - Flash Attention 2 replaces the standard attention computation with a memory-efficient kernel that avoids materialising the full N×N attention matrix. For long sequences (up to 4096 tokens here), standard attention memory grows quadratically with sequence length; Flash Attention keeps it linear by
  computing attention in tiles that stay in fast SRAM. The result is both lower peak memory and significantly faster throughput.
  - DeepSpeed (ZeRO optimizer) shards optimizer states, gradients, and optionally model parameters across all GPUs on a node rather than replicating them. Even with LoRA reducing trainable parameters, the frozen base model weights (7B × bf16 = ~14GB) still need to live somewhere. ZeRO stage 2 or 3
  distributes this across GPUs, enabling larger effective batch sizes and leaving more headroom for activations.

  Together they make a training run that would otherwise require dozens of GPUs in full fine-tuning mode feasible on 2 nodes of 8 H100s, while still converging to a high-quality model.
