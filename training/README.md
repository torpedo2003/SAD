# Training and evaluation

This folder trains a SAD decoder and evaluates it. The walkthrough uses DiT-XL/2 on ImageNet 256×256;
[Your own generator](#your-own-generator) shows what to change for another model.
All commands are run from this folder.

One 24 GB GPU is enough. Training takes about one GPU-hour. Most of the compute goes into caching generated latents,
which is done once per generator.

## 1. Setup

```bash
conda create -n sad python=3.11 -y && conda activate sad
pip install -r requirements.txt pytorch-fid
export PYTHONPATH=$PWD

mkdir -p data && cd data
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
huggingface-cli download benjamin-paine/imagenet-1k-256x256 --repo-type dataset --include "data/train-*" --local-dir hf_in256
ln -s hf_in256/data imagenet-1k-256x256 && cd ..
python prepare_data.py
```

The data has two parts:

- **ADM reference batch (10k images).** Its split A (5k images) is the real target of GSA. Its split B is held out for evaluation.
- **ImageNet images.** EAP reconstructs them through the encoder.

## 2. Cache generated latents

Sample the frozen generator once. Store the latents in raw VAE space, i.e. before multiplying by the scaling factor.

```bash
for r in 0 1 2 3 4 5 6 7; do   # training pool: 8 × 5k latents (16k is already enough)
  python gen_latents.py --n 5000 --cfg 1.5 --seed $((1000+r)) --out data/gen_train/cfg1.5_r$r.npz
done
for r in 0 1 2 3 4 5 6 7; do   # evaluation set: 8 × 1,250 latents, with different seeds
  python gen_latents.py --n 1250 --cfg 1.5 --seed $((100+r)) --out data/gen/all_cfg1.5_r$r.npz
done
```

The shards are independent, so you can run them on several GPUs in parallel.

## 3. Train

```bash
python train_sad.py --mode gen --loss gan --lam 0.05 --gen_pool cfg1.5 --out runs/sad_gan
```

`runs/sad_gan/decoder.pt` contains only the decoder weights. Load it with `load_sad_decoder(vae, local_path="training/runs/sad_gan/decoder.pt")` from the
repository root. `log.jsonl` logs the EAP loss (`anchor`), the GSA loss (`dist`) and the discriminator loss (`d`).

| Flag | Meaning | Default |
|---|---|---|
| `--loss` | GSA discrepancy in DINOv2-S space: `gan`, `mmd` or `fd` | `fd` |
| `--lam` | GSA weight λ. We use 0.05 for `gan` and 1 for `mmd`; lower it if colors drift | 1.0 |
| `--lam_anchor` | EAP weight | 1.0 |
| `--gen_pool` | reads `data/gen_train/<pool>_r*.npz`; separate several pools with commas | `cfg1.5` |
| `--target_npz` | your own real images for GSA (`images`: uint8, N×H×W×3) | ADM split A |
| `--anchor_npz` | your own images for EAP, same format | ImageNet |
| `--steps`, `--lr`, `--bs`, `--bs_anchor` | 2,000 steps, AdamW 1e-5, batch 16 (GSA) + 8 (EAP) | as listed |

## 4. Evaluate

```bash
python eval_decoder.py --sets all_cfg1.5 --out results/d0.json                                 # original decoder
python eval_decoder.py --sets all_cfg1.5 --ckpt runs/sad_gan/decoder.pt --out results/sad_gan.json
```

The script computes FID, FD<sub>CLIP</sub>, precision and recall on the 10k generated samples against split B. It also
computes rFID and PSNR on real images, which check reconstruction. With the commands above, expect FID 13.32 → 12.36,
FD<sub>CLIP</sub> 51.4 → 42.4 and rFID 2.56.

To run the official ADM evaluator (FID-50K, IS), export the decoded samples:

```bash
python dump_samples.py --set all_cfg1.5 --name sad_gan --ckpt runs/sad_gan/decoder.pt   # -> data/samples/all_cfg1.5__sad_gan.npz
```

Then pass the file to `evaluations/evaluator.py` from [guided-diffusion](https://github.com/openai/guided-diffusion).
For FID-50K, cache 50k latents in step 2 first.

## Your own generator

SAD needs three things: the frozen generator's latents, a few thousand real images from the target distribution, and the
autoencoder. The generator code never enters training.

1. **Latents.** Write a script that samples your generator and saves `np.savez(path, z=latents)` into
   `data/gen_train/<name>_r<k>.npz` (float, N×C×h×w, raw VAE space). Use the sampler and guidance you deploy with: a
   decoder is specific to them. 16k–40k latents are enough. `gen_latents_*.py` are examples for DiT, SiT, LightningDiT,
   REPA-E and PixArt-α.
2. **Real images.** Save a few thousand unpaired images as `np.savez(path, images=uint8 N×H×W×3)`. They do not need to
   match the latents. For text-to-image, images from the same caption domain work well (we used COCO for PixArt-α).
3. **Autoencoder.** `load_vae()` in `vae_utils.py` returns the model to fine-tune. It must provide `.decode(z).sample` and
   contain `decoder.*` / `post_quant_conv.*` weights. SD-VAE works as is; `SAD_VAE=vavae` and `SAD_VAE=e2evavae` select
   the two other tokenizers of the paper. For another autoencoder, add a branch there.

Then train:

```bash
python train_sad.py --mode gen --loss gan --lam 0.05 --gen_pool <name> \
    --target_npz data/my_real.npz --anchor_npz data/my_real.npz --out runs/my_sad
```

**Tips:**

- Start with `gan` and λ = 0.05. If colors or textures drift away from the original decoder, lower λ; we used 0.01 for PixArt-α.
- `mmd` (λ = 1) is more conservative and has no discriminator.
- Compare against the original decoder on held-out latents, i.e. sampled with other seeds. `dump_samples.py` works with
  any `data/gen/<set>_r*.npz`.
