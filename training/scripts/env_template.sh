# Copy to env.sh and edit. All scripts assume training/ as the working directory.
export PYTHONPATH=$PWD
export HF_HOME=/path/to/hf_cache          # Hugging Face cache (DiT VAE, DINOv2, CLIP, SigLIP, PickScore)
export PY=python
# Expected relative layout (create symlinks as needed):
#   data/imagenet-1k-256x256/   ImageNet-1k 256x256 parquet shards (anchor images for EAP)
#   data/VIRTUAL_imagenet256_labeled.npz   ADM reference batch; data/adm_split.npz: random split A (GSA target) / B (evaluation)
#   data/gen_train/<pool>_r*.npz           cached generated latents for training (gen_latents*.py)
#   data/gen/<set>_r*.npz                  generated latents for evaluation (seeds disjoint from training)
#   pretrained/                            released generator / tokenizer checkpoints (see README)
#   third_party/                           official code of DiT, SiT, REPA, REPA-E, LightningDiT, DG
