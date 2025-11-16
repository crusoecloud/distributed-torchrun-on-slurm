#!/usr/bin/env python3
"""
GPU Burn-in Test for H200 GPUs
Runs intensive training on a large model for 4 hours
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import os
import socket
from datetime import datetime, timedelta


def setup_distributed():
    """Initialize distributed training environment"""
    # Slurm environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def log_system_info(rank, local_rank):
    """Log system and GPU information"""
    if rank == 0:
        print("="*60)
        print("GPU BURN-IN TEST STARTED")
        print("="*60)
    
    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    
    print(f"Host: {hostname}, Local GPU: {local_rank}, Device: {gpu_name}, Memory: {gpu_memory:.1f}GB")

def create_synthetic_batch(batch_size, seq_length, vocab_size, device):
    """Create synthetic training data"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def monitor_gpu_stats(local_rank):
    """Monitor and log GPU statistics"""
    mem_allocated = torch.cuda.memory_allocated(local_rank) / 1e9
    mem_reserved = torch.cuda.memory_reserved(local_rank) / 1e9
    max_mem = torch.cuda.max_memory_allocated(local_rank) / 1e9
    
    # Get GPU utilization if possible
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        
        return {
            'mem_allocated_gb': mem_allocated,
            'mem_reserved_gb': mem_reserved,
            'max_mem_gb': max_mem,
            'gpu_util_percent': util.gpu,
            'mem_util_percent': util.memory,
            'temp_c': temp,
            'power_w': power
        }
    except:
        return {
            'mem_allocated_gb': mem_allocated,
            'mem_reserved_gb': mem_reserved,
            'max_mem_gb': max_mem
        }

def run_burnin_test(duration_hours=4):
    """Main burn-in test function"""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Log system info
    log_system_info(rank, local_rank)
    
    # Model configuration - using a large model (Llama-2-70B style)
    if rank == 0:
        print("Loading model configuration...")
    
    # You can change this to any large model from HuggingFace
    model_name = "meta-llama/Llama-2-70b-hf"  # Replace with available model
    
    try:
        config = AutoConfig.from_pretrained(model_name)
    except:
        # Fallback: create a large config manually
        from transformers import LlamaConfig
        config = LlamaConfig(
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=48,
            num_attention_heads=40,
            num_key_value_heads=8,
            vocab_size=32000,
            max_position_embeddings=4096,
        )
        if rank == 0:
            print("Using manual config for large model")
    
    # Load model
    if rank == 0:
        print("Initializing model (this may take a few minutes)...")
    
    torch.cuda.empty_cache()
    
    # Create model with bf16 for H200
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(torch.bfloat16)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training parameters
    batch_size = 4  # Adjust based on memory
    seq_length = 2048
    vocab_size = config.vocab_size
    
    if rank == 0:
        print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
        print(f"Starting {duration_hours}-hour burn-in test...")
        print("="*60)
    
    # Burn-in test loop
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    iteration = 0
    total_tokens = 0
    log_interval = 10
    
    try:
        while time.time() < end_time:
            iter_start = time.time()
            
            # Create synthetic batch
            batch = create_synthetic_batch(batch_size, seq_length, vocab_size, device)
            
            # Forward pass
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            iter_time = time.time() - iter_start
            tokens_per_sec = (batch_size * seq_length) / iter_time
            total_tokens += batch_size * seq_length
            
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remaining_str = str(timedelta(seconds=int(remaining)))
                
                stats = monitor_gpu_stats(local_rank)
                stats_str = ", ".join([f"{k}: {v:.2f}" for k, v in stats.items()])
                
                print (
                    f"Iter {iteration} | Loss: {loss.item():.4f} | "
                    f"Tokens/sec: {tokens_per_sec:.0f} | "
                    f"Elapsed: {elapsed_str} | Remaining: {remaining_str} | "
                    f"{stats_str}"
                )
            
            iteration += 1
            
            # Synchronize periodically to detect any GPU failures
            if iteration % 100 == 0:
                dist.barrier()
        
        # Test completed successfully
        total_time = time.time() - start_time
        if rank == 0:
            print("="*60)
            print("BURN-IN TEST COMPLETED SUCCESSFULLY")
            print(f"Total time: {timedelta(seconds=int(total_time))}")
            print(f"Total iterations: {iteration}")
            print(f"Total tokens processed: {total_tokens:,}")
            print("="*60)
        
    except Exception as e:
        print(f"BURN-IN TEST FAILED: {str(e)}")
        raise
    
    finally:
        # Cleanup
        dist.destroy_process_group()

if __name__ == "__main__":
    run_burnin_test(duration_hours=4)

