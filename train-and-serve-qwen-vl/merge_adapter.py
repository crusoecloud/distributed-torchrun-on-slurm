# Take the name of the output dir created by your training run (where the LoRA fine tuned model is saved)
# Say it is in ./outputs/qwen25vl-sft-42, then:
# OUTPUT_SUBDIR_NAME=qwen25vl-sft-42 python merge_adapter.py
# This creates a new dir for the merged model (under ./models) that you can then use with serve_model.sbatch

from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
import os

OUTPUT_SUBDIR_NAME  = os.environ.get("OUTPUT_SUBDIR_NAME")

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      "./models/qwen25vl-7b", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, "./outputs/" + OUTPUT_SUBDIR_NAME)
merged = model.merge_and_unload()
merged.save_pretrained("./models/" + OUTPUT_SUBDIR_NAME)

# Also copy the processor/tokenizer files
from transformers import AutoProcessor
AutoProcessor.from_pretrained("./models/qwen25vl-7b").save_pretrained("./models/" + OUTPUT_SUBDIR_NAME)