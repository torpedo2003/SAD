# Training and evaluation

This folder contains everything used in the paper. All commands are run from this folder.
The walkthrough below trains and evaluates SAD for **DiT-XL/2 with 50-step DPM-Solver++** (the setting of our ablations);
other generators follow the same three steps and are listed at the end.

**Hardware.** One GPU with 24 GB is enough for training (batch 16 + 8). Caching latents is the only expensive step
(about 8 GPU-hours for 40k DiT-XL/2 latents) and is done once per generator.

## 0. Environment

```bash
conda create -n sad python=3.11 -y && conda activate sad
pip install -r requirements.txt
pip install pytorch-fid            # Inception features for the 10k-sample FID used in ablations
export PYTHONPATH=$PWD:$PWD/analysis
```

## 1. Data

| What | Where it goes | Used for |
|---|---|---|
| ADM ImageNet-256 reference batch [`VIRTUAL_imagenet256_labeled.npz`](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) | `data/` | real images for GSA (split A) and evaluation (split B, FID-50K statistics) |
| ImageNet-1k 256×256 parquet shards ([`benjamin-paine/imagenet-1k-256x256`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256), `train-*.parquet`) | `data/imagenet-1k-256x256/` | encoder latents for EAP, reconstruction metrics |

```bash
mkdir -p data && cd data
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
huggingface-cli download benjamin-paine/imagenet-1k-256x256 --repo-type dataset --include "data/train-*" --local-dir hf_in256
ln -s hf_in256/data imagenet-1k-256x256
cd ..
python prepare_data.py     # -> data/adm_split.npz (A/B split, identical to the paper) and data/real75.npz (10k images for rFID/PSNR)
```

GSA uses only the 5k images of split A as real images. Split B is never seen in training and is the reference for every
held-out metric.

## 2. Cache generated latents (once per generator)

The generator is only sampled, never trained. Training and evaluation latents use **different seeds**.

```bash
# training pool: 8 x 5,000 = 40k latents, 50-step DPM-Solver++, guidance 1.5
for r in 0 1 2 3 4 5 6 7; do
  python gen_latents.py --n 5000 --cfg 1.5 --seed $((1000+r)) --out data/gen_train/cfg1.5_r$r.npz
done
# evaluation set: 8 x 1,250 = 10k latents (ablation protocol)
for r in 0 1 2 3 4 5 6 7; do
  python gen_latents.py --n 1250 --cfg 1.5 --seed $((100+r)) --out data/gen/all_cfg1.5_r$r.npz
done
# optional, for FID-50K: 8 x 6,250 = 50k latents
for r in 0 1 2 3 4 5 6 7; do
  python gen_latents.py --n 6250 --cfg 1.5 --seed $((900+r)) --out data/gen/dit50k_cfg1.5_r$r.npz
done
```

Each shard is independent, so the loops can run in parallel on several GPUs (`CUDA_VISIBLE_DEVICES=$r ... &`).
16k training latents are already enough for the 50-step sampler; 40k is what we use by default.

## 3. Train SAD

```bash
python train_sad.py --mode gen --loss gan --lam 0.05 --steps 2000 --gen_pool cfg1.5 --out runs/sad_gan
python train_sad.py --mode gen --loss mmd --lam 1    --steps 2000 --gen_pool cfg1.5 --out runs/sad_mmd
```

About one GPU-hour each. The output folder contains `decoder.pt` (decoder + post-quant conv), `args.json` and `log.jsonl`
(every 50 steps: EAP loss `anchor`, GSA loss `dist`, discriminator loss `d`).

| Flag | Meaning | Default |
|---|---|---|
| `--mode gen` | GSA on cached generated latents (SAD). `enc` = GSA on encoder latents (control), `none` = anchor only | – |
| `--loss` | GSA discrepancy in DINOv2-S space: `gan`, `mmd` or `fd` | `fd` |
| `--lam` | weight λ of GSA (0.05 for GAN, 1 for MMD on ImageNet) | 1.0 |
| `--lam_anchor` | weight of EAP (0 removes the anchor) | 1.0 |
| `--gen_pool` | `data/gen_train/<pool>_r*.npz`, comma-separated pools are concatenated | `cfg1.5` |
| `--steps`, `--lr`, `--bs`, `--bs_anchor` | 2,000 steps, AdamW 1e-5, batch 16 (GSA) + 8 (EAP) | as listed |
| `--target_npz`, `--anchor_npz` | custom real images for GSA / EAP (`images`, uint8 NHWC) | ADM split A / ImageNet |
| `--n_real`, `--n_gen` | use only the first n real images / cached latents (data-efficiency study) | all |

## 4. Evaluate

**10k-sample metrics (ablation protocol).** FID and FD<sub>CLIP</sub> against split B, plus rFID and PSNR on 10k real images:

```bash
python eval_decoder.py --sets all_cfg1.5 --out results/d0.json                                   # original decoder D_0
python eval_decoder.py --sets all_cfg1.5 --ckpt runs/sad_gan/decoder.pt --out results/sad_gan.json
```

The first call also caches the reference features in `data/proc/`. The JSON holds `all_cfg1.5 → {fid, fd_clip, precision, recall}`
and `recon → {rfid, psnr}`. Expected: D<sub>0</sub> FID 13.32, FD<sub>CLIP</sub> 51.4; SAD-GAN FID 12.36, FD<sub>CLIP</sub> 42.4, rFID 2.56.

**FID-50K with the official ADM evaluator** (the protocol of Tab. 2 in the paper):

```bash
python dump_samples.py --set dit50k_cfg1.5 --name d0
python dump_samples.py --set dit50k_cfg1.5 --name sad_gan --ckpt runs/sad_gan/decoder.pt
git clone https://github.com/openai/guided-diffusion && cd guided-diffusion/evaluations
pip install -r requirements.txt                                     # TensorFlow, used only by the evaluator
python evaluator.py ../../data/VIRTUAL_imagenet256_labeled.npz ../../data/samples/dit50k_cfg1.5__sad_gan.npz
```

`fid50k.py` gives a quick PyTorch estimate of the same numbers (plus FD<sub>CLIP</sub> against split B) without TensorFlow:

```bash
python fid50k.py --ckpt runs/sad_gan/decoder.pt --set dit50k_cfg1.5 --ref_split_b --out results/fid50k_sad_gan.json
```

**Metrics unused in training** (KID, FD in SigLIP space, CMMD, HPS v2.1, PickScore):

```bash
python eval_metrics.py imagenet orig: gen_gan0.05:runs/sad_gan/decoder.pt
```

## 5. Other generators

The recipe is identical; only the latent cache (and, for VA-VAE / E2E-VAE, the autoencoder) changes.

| Generator | Cache latents with | Autoencoder (`SAD_VAE=`) | Train |
|---|---|---|---|
| DiT-XL/2, official 250-step DDPM | `gen_latents_dit_official.py` (official DiT code in `third_party/DiT`) | SD-VAE (default) | `--gen_pool ditoff40` |
| SiT-XL/2 + REPA | `gen_latents_repa.py` | SD-VAE (default) | `--gen_pool repa` |
| LightningDiT-XL | `gen_latents_ldit.py` (official code in `third_party/LightningDiT`) | `vavae` | `--gen_pool ldit` |
| REPA-E | `gen_latents_repae.py` | `e2evavae` | `--gen_pool repae` |
| PixArt-α 256px | `gen_latents_pixart.py` (COCO captions) | SD-VAE (default) | `--lam 0.01 --target_npz <COCO images>` |
| DiT-XL/2 + discriminator guidance | `gen_latents_dg.py` (official DG discriminator) | SD-VAE (default) | evaluate the DiT decoder on DG latents |

For example, LightningDiT:

```bash
export SAD_VAE=vavae
python gen_latents_ldit.py --seed 1000 --out data/gen_train/ldit_r0.npz   # repeat for r = 0..7
python train_sad.py --mode gen --loss gan --lam 0.05 --gen_pool ldit --out runs/ldit_sad_gan
```

Checkpoint paths of the released generators are set at the top of each `gen_latents_*.py` and `vavae_utils.py`
(`pretrained/...`); download the official weights from the respective repositories.

## 6. Baselines and analyses

| Command | Paper |
|---|---|
| `python train_sad.py --mode noise --noise 0.1 --out runs/base_noise` | noise-robust decoder (Tab. 4) |
| `python gen_sdedit.py --t_start 250 --out data/sdedit/t250` then `python train_sad.py --mode paired --paired_data data/sdedit/t250 --out runs/base_paired` | paired post-training (Tab. 4) |
| `python train_sad.py --mode gen --arch refiner --loss gan --lam 0.05 --out runs/base_refiner` | pixel refiner after D<sub>0</sub> (Tab. 4) |
| `python train_fdda.py --help` | FDDA re-implementation (Tab. 4, Tab. 9) |
| `python analysis/e3_baselines.py` | public decoders: sd-vae-ft-mse, Consistency Decoder (Tab. 4) |
| `python train_sad.py --mode enc --loss gan --lam 0.05 --out runs/abl_enc` | GSA on encoder latents (Tab. 6, Fig. 5) |
| `python train_sad.py --mode gen --loss gan --lam 0.05 --lam_anchor 0 --out runs/abl_noanc` | without EAP (Tab. 6) |
| `python g0_process.py --set all_cfg1.5` then `python g0_analysis.py` | off-manifold test (Tab. 14) |
| `python analysis/exp_color_stats.py 3000` | color and exposure statistics (Tab. 8) |
| `python analysis/make_tables.py` | rebuilds every table of the paper from `results/` |

## File overview

| File | Content |
|---|---|
| `train_sad.py` | SAD training (GSA + EAP, Algorithm 1), ablations and decoder baselines |
| `common.py` | DINOv2-S features, discriminator heads, data loaders |
| `vae_utils.py`, `vavae_utils.py` | SD-VAE / VA-VAE / E2E-VAE wrappers |
| `refiner.py` | loads a fine-tuned decoder (or the pixel refiner) in place of D<sub>0</sub> |
| `gen_latents*.py`, `gen_sdedit.py`, `encode_real.py` | latent caching and real-image encoding |
| `prepare_data.py`, `dump_samples.py` | data preparation and ADM-evaluator sample export |
| `eval_decoder.py`, `fid.py`, `fid50k.py`, `eval_metrics.py`, `pick_eval.py`, `fdclip_images.py` | evaluation |
| `train_fdda.py`, `analysis/` | baselines, analyses and table generation |
