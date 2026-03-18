"""
Serve locally-downloaded Qwen2.5-VL-7B-Instruct with vLLM.

Usage:
    MODEL_PATH=./models/qwen25vl-7b python serve_model.py

Optional env vars:
    MODEL_PATH  - path to downloaded model (default: ./models/qwen25vl-7b)
    HOST        - bind host (default: 0.0.0.0)
    PORT        - bind port (default: 8000)
    GPU_COUNT   - number of GPUs for tensor parallelism (default: 1)
    MAX_MODEL_LEN - max context length in tokens (default: 8192)

The server exposes:
    GET  /v1/models                  - list available models
    POST /v1/chat/completions        - chat completions (text + image)

See CURL_EXAMPLES.md (printed to stdout on startup) for request examples.
"""

import os
import subprocess
import sys

MODEL_PATH    = os.environ.get("MODEL_PATH", "./models/qwen25vl-7b")
HOST          = os.environ.get("HOST", "0.0.0.0")
PORT          = os.environ.get("PORT", "8000")
GPU_COUNT     = os.environ.get("GPU_COUNT", "1")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
NODE_HOSTNAME = os.environ.get("NODE_HOSTNAME")

CURL_EXAMPLES = f"""
*****
1. See details of models served
curl http://{NODE_HOSTNAME}:{PORT}/v1/models/

2. Ask the model to describe an image from Wikimedia
curl http://{NODE_HOSTNAME}:{PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{MODEL_PATH}",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{
            "type": "image_url",
            "image_url": {{
              "url": "https://upload.wikimedia.org/wikipedia/commons/7/75/Supertanker_AbQaiq.jpg"
            }}
          }},
          {{
            "type": "text",
            "text": "Describe this image in detail."
          }}
        ]
      }}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }}'
  *****
  """

def check_model_present(path):
    config = os.path.join(path, "config.json")
    if not os.path.isfile(config):
        print(
            f"ERROR: model not found at '{path}'. "
            "Run download_model.py first (or set MODEL_PATH).",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    check_model_present(MODEL_PATH)

    print(CURL_EXAMPLES)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", GPU_COUNT,
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--limit-mm-per-prompt", "{\"image\": 5}",  # allow up to 5 images per request
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
