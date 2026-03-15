#!/bin/bash

uv --python 3.12 venv
source .venv/bin/activate
uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install transformers deepspeed accelerate webdataset qwen-vl-utils sorted peft
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu128
