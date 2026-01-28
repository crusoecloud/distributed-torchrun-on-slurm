# Running TorchTitan on Crusoe Cloud's SLURM Solution #

From the TorchTitan repo (https://github.com/pytorch/torchtitan):

*Torchtitan is a PyTorch native platform designed for rapid experimentation and large-scale training of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, torchtitan provides a flexible foundation for developers to build upon. With torchtitan extension points, one can easily create custom extensions tailored to specific needs.*

We will do multinode fine tuning of Llama3_8b using the C4 dataset. These instructions are specific to GB200, but the general principle is the same for any Crusoe Slurm cluster. The results section at the bottom of the page will be extended to cover multiple GPUs and cluster and batch sizes.  

Clone the torchtitan repo to the home dir of the cluster under test.  

On any one of the compute nodes (not the login or head node!) create a Python virtual environment, activate it, and install the requirements:
```
sudo apt-get install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu130
#Download tokenizer from HuggingFace
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=hf_xxxxxxxxxxxx
```
Update multinode_trainer.slurm for the correct number of nodes and gpus per node (in both the ‘sbatch’ parts and in the torchrun command near the bottom) and add a line to activate the venv.

** For quicker performance on repeated training runs: download C4 data set to cluster's /data volume **
```
cd /data
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/datasets/allenai/c4
cd c4
#this next step takes about an hour
git lfs pull --include "en/*"
```
...and update the .toml training config file with * dataset_path = "/data/c4/" * in the \[training\] section
                                                                                                                                         
### Running the job ###

Run the job from the slurm login node: sbatch multinode_trainer.sbatch. ‘Watch squeue’ to see that the job is running - if it doesn’t go to ‘R’ status it could be that you didn’t have sufficient nodes in ‘idle’ state, or that you request resources that no node has (e.g too many GPU or CPU per node)
When the job is running, its logs will be written to slurm-x.out where x is the ID of the slurm job. The outputs below show the performance to be expected on a cluster where all the nodes are in the same imex partition.



# TorchTitan performance results for various cluster configs #

|GPU Type|Compute nodes in training job|Batch size|Performance indicators at step 1000 of Torchtitan test described on this page|
|--------|-----------------------------|----------|-----------------------------------------------------------------------------|
|GB200   |17 (68 GPU)                  |5         |step: 100  loss:  6.4277  grad_norm:  2.9062  memory: 158.61GiB(86.20%)  tps: 17,581  tflops: 1,018.20  mfu: 45.25%|
|GB200   |9 (36 GPU)                   |5         |step: 100  loss:  8.2954  grad_norm: 29.3788  memory: 159.31GiB(86.58%)  tps: 17,975  tflops: 1,040.99  mfu: 46.27%|
|GB200   |3 (12 GPU)                   |5         |step: 100  loss:  6.5572  grad_norm:  3.6680  memory: 161.65GiB(87.85%)  tps: 18,284  tflops: 1,058.93  mfu: 47.06%|
|B200    |2 (16 GPU)                   |5         |step: 10  loss:  9.9256  grad_norm:  9.2181  memory: 161.51GiB(90.56%)  tps: 17,441  tflops: 1,010.08  mfu: 44.89%|


