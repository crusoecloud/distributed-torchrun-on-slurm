"""
Download Qwen/Qwen2.5-VL-7B-Instruct from HuggingFace to a local directory.
Skips the download if the model is already present.
HF_TOKEN must be set as an environment variable.
"""
import os
import sys
from huggingface_hub import snapshot_download

HF_REPO    = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_TOKEN   = os.environ.get("HF_TOKEN")
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/qwen25vl-7b")


def _model_present(path):
    return os.path.isfile(os.path.join(path, "config.json"))


if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

if _model_present(MODEL_PATH):
    print(f"Model already present at {MODEL_PATH}, skipping download.")
else:
    print(f"Downloading {HF_REPO} to {MODEL_PATH}...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, local_dir=MODEL_PATH, token=HF_TOKEN)
    print(f"Download complete.")
