#!/bin/bash
# Minimal SAD pipeline for DiT-XL/2 (SD-VAE). Run from training/ after `source scripts/env.sh`; see README.md.
set -e

# 1) Cache generated latents (training pool and a held-out evaluation set with different seeds)
for r in 0 1 2 3 4 5 6 7; do
  $PY gen_latents.py --n 5000 --cfg 1.5 --seed $((1000+r)) --out data/gen_train/cfg1.5_r$r.npz
  $PY gen_latents.py --n 1250 --cfg 1.5 --seed $((100+r))  --out data/gen/all_cfg1.5_r$r.npz
done

# 2) Train SAD (about one GPU-hour)
$PY train_sad.py --mode gen --loss gan --lam 0.05 --gen_pool cfg1.5 --out runs/sad_gan

# 3) Evaluate against the original decoder
$PY eval_decoder.py --sets all_cfg1.5 --out results/d0.json
$PY eval_decoder.py --sets all_cfg1.5 --ckpt runs/sad_gan/decoder.pt --out results/sad_gan.json
