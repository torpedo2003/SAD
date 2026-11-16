# Code

All scripts are run from this directory (`source scripts/env.sh` first). Paths to data and checkpoints are relative; see
`scripts/env_template.sh` for the expected layout.

## SAD (Sec. 3)

| File | Content |
|---|---|
| `train_sad.py` | Training of SAD: GSA (`--loss gan/mmd/fd`, Eqs. 3), EAP (Eq. 4), objective (Eq. 5), Algorithm 1. `--mode gen` aligns generated latents (SAD); `--mode enc` is the GSA-on-E(x) control; `--mode none`, `--lam_anchor 0` and the other modes give the ablations and baselines. |
| `common.py` | Frozen DINOv2-S feature network, multi-block discriminator heads, data loaders for real images and cached latents. |
| `vae_utils.py`, `vavae_utils.py` | Encoder/decoder wrappers for SD-VAE, VA-VAE and E2E-VAE. |
| `refiner.py` | Loads a fine-tuned decoder (or the pixel-refiner baseline) in place of `D_0`. |

## Latent caching (generators are only sampled, never trained)

`gen_latents.py` (DiT-XL/2, DPM-Solver++ / DDPM), `gen_latents_dit_official.py` (official DiT code, 250-step DDPM),
`gen_latents_sit.py`, `gen_latents_sit_sde.py`, `gen_latents_repa.py` (SiT-XL/2 + REPA), `gen_latents_ldit.py` (LightningDiT-XL),
`gen_latents_repae.py` (REPA-E), `gen_latents_pixart.py` (PixArt-alpha), `gen_latents_dg.py` (discriminator guidance),
`gen_sdedit.py` (SDEdit pairs for the paired post-training baseline), `encode_real.py` (encoder latents of real images).

## Evaluation (Sec. 4)

| File | Content |
|---|---|
| `fid.py`, `fid50k.py` | ADM-evaluator FID-50K, sFID, IS, precision/recall; FD_CLIP against the held-out split. |
| `eval_decoder.py` | 10k-sample FID, FD_CLIP, rFID and PSNR used in the ablations. |
| `eval_metrics.py`, `pick_eval.py`, `fdclip_images.py` | Held-out metrics: KID, FD in SigLIP space, CMMD, HPS v2.1, PickScore. |
| `g0_analysis.py`, `g0_process.py` | Off-manifold test (Tab. 14). |

## Baselines and analysis

| File | Content |
|---|---|
| `train_fdda.py` | Re-implementation of FDDA (Tab. 4, Tab. 9). |
| `analysis/e3_baselines.py` | Public decoders (sd-vae-ft-mse, Consistency Decoder). |
| `analysis/exp_color_stats.py` | Color and exposure statistics (Tab. 8). |
| `analysis/make_tables.py` | Builds every table body of the paper from `../results/metrics/`. |
| `analysis/make_qual_diff.py` | Per-pixel difference maps between the two decoders. |
