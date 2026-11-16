#!/bin/bash
# Main commands behind the paper results (DiT-XL/2, SD-VAE). Run from training/ after `source scripts/env.sh`.
set -e

# 1) Cache generated latents once (seeds disjoint from evaluation).
#    DiT-XL/2, 50-step DPM-Solver++, guidance 1.5: training pool and 10k evaluation set
$PY gen_latents.py --help
#    Official DiT recipe (250-step DDPM): gen_latents_dit_official.py; LightningDiT: gen_latents_ldit.py;
#    SiT-XL/2 + REPA: gen_latents_repa.py; REPA-E: gen_latents_repae.py; PixArt-alpha: gen_latents_pixart.py;
#    discriminator guidance: gen_latents_dg.py

# 2) Train SAD (GSA with the GAN discrepancy + EAP), about one GPU-hour
$PY train_sad.py --mode gen --loss gan --lam 0.05 --steps 2000 --gen_pool cfg1.5 --out runs/sad_gan
#    SAD-MMD
$PY train_sad.py --mode gen --loss mmd --lam 1 --steps 2000 --gen_pool cfg1.5 --out runs/sad_mmd
#    Ablations (Tab. 6): GSA on encoder latents / without EAP / without GSA
$PY train_sad.py --mode enc  --loss gan --lam 0.05 --out runs/abl_gsa_on_enc
$PY train_sad.py --mode gen  --loss gan --lam 0.05 --lam_anchor 0 --out runs/abl_no_eap
$PY train_sad.py --mode none --out runs/abl_no_gsa
#    Baselines (Tab. 4): noise-robust decoder, paired post-training, pixel refiner, FDDA re-implementation
$PY train_sad.py --mode noise --noise 0.1 --out runs/base_noise
$PY train_sad.py --mode paired --paired_data data/sdedit/t250 --out runs/base_paired    # pairs from gen_sdedit.py
$PY train_sad.py --mode gen --arch refiner --loss gan --lam 0.05 --out runs/base_refiner
$PY train_fdda.py --help

# 3) Evaluate (FID / FD_CLIP / P&R against the held-out reference split B, rFID / PSNR)
$PY eval_decoder.py --ckpt runs/sad_gan/decoder.pt --out results/sad_gan.json
$PY fid50k.py --ckpt runs/sad_gan/decoder.pt --set dit50k_cfg1.5 --ref_split_b --out results/fid50k_sad_gan.json
#    Held-out metrics (KID, FD in SigLIP space, CMMD, HPS v2.1, PickScore)
$PY eval_metrics.py imagenet orig: gen_gan0.05:runs/sad_gan/decoder.pt

# 4) Analyses: off-manifold test (Tab. 14), color statistics (Tab. 7), all paper tables from results/metrics
$PY g0_analysis.py --help
$PY analysis/exp_color_stats.py 3000
$PY analysis/make_tables.py
