import os
import glob
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import webdataset as wds
from PIL import Image
import io
# --- CONFIGURATION (Update these with your shared drive paths) ---
MODEL_PATH=os.getenv('MODEL_PATH','./models/qwen25vl-7b')
DATA_PATH=os.getenv('DATA_PATH','./data/llava_webdataset/pretrain*.tar')
OUTPUT_DIR=os.getenv('OUTPUT_DIR')
# -----------------------------------------------------------------

def parse_webdataset(sample):
    """
    Parses a single sample from the WebDataset.
    Format: img0.jpg..imgN.jpg + json with {"captions": [...]} list.
    Yields one messages dict per valid (image, caption) pair.
    """
    try:
        json_data = sample.get("json")
        if json_data is None:
            return
        meta = json.loads(json_data)
        captions = meta.get("captions", [])
        for i, caption in enumerate(captions):
            img_key = f"img{i}.jpg"
            if img_key not in sample:
                continue
            try:
                image = Image.open(io.BytesIO(sample[img_key])).convert("RGB")
            except Exception:
                continue
            print(f"[parse_webdataset] {img_key}: {caption[:200]}", flush=True)
            yield {"messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": "Describe this image in detail."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                },
            ]}
    except Exception:
        pass

@dataclass
class QwenDataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for feature in features:
            messages = feature["messages"]
            # Apply chat template for text structure
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            
            # Extract PIL images from messages
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image":
                        images.append(content["image"])

        # Process everything into tensors
        if not images:
            raise ValueError("Batch contains no images — check parse_webdataset and dataset pipeline")
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            max_pixels=1048576,
        )
        
        # Create labels for causal LM training (shift labels inside model)
        # We mask padding tokens to -100 so they aren't included in loss
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Setting up WebDataset pipeline...")
    shard_files = sorted(glob.glob(DATA_PATH))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {DATA_PATH}")

    rank       = int(os.environ.get("RANK",       "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_shards = shard_files[rank::world_size]
    print(f"[Rank {rank}] Using {len(node_shards)}/{len(shard_files)} shards")

    dataset = (
        wds.WebDataset(node_shards, shardshuffle=100, nodesplitter=None)
        .repeat()
        .compose(wds.split_by_worker)
        .compose(lambda src: (item for sample in src for item in parse_webdataset(sample)))
    )
    collator = QwenDataCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Adjust based on VRAM (H100 80GB can likely handle 4-8)
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,
        max_steps=5000, # IterableDatasets usually use max_steps instead of epochs
        logging_steps=10,
        save_steps=500,
        gradient_checkpointing=True,
        deepspeed="deepspeed_config.json", # Point to DeepSpeed config
        report_to="none", # Change to "wandb" if you use Weights & Biases
        remove_unused_columns=False, # Crucial for custom multimodal data
        ddp_broadcast_buffers=False,
        accelerator_config={"dispatch_batches": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
