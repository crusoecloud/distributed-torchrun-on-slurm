from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='lmms-lab/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M',
    repo_type='dataset',
    local_dir='./data/llava_webdataset',
    allow_patterns=['*.tar'],
    local_dir_use_symlinks=False,
)
