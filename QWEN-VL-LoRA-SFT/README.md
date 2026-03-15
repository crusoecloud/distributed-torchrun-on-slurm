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
