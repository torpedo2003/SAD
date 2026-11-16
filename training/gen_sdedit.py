"""Paired data for the RobusTok-style baseline (Qiu et al. 2025, 'Image Tokenizer Needs Post-Training',
diffusion variant via SDEdit): take a real image x with its label, encode z0 = E(x), noise it to timestep
t_start (preservation ratio sigma = 1 - t_start/T), denoise with DiT (DDIM, cfg) back to z_sd.
The decoder is then trained to map z_sd -> D_0(z0) (paired), see train_sad.py --mode paired.

Saves z0 (raw VAE latent of the real image), z_sd (raw), y.
"""

import argparse
import glob
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers import DDIMScheduler, DiTPipeline

from common import SHARDS, TRAIN_SHARDS
from vae_utils import decode_jpeg, from_uint8


def real_with_labels(n, seed):
    """n images, 2 per random row group (row groups are single-class), with labels."""
    rng = random.Random(seed)
    pfs = [pq.ParquetFile(f) for s in TRAIN_SHARDS for f in glob.glob(SHARDS.format(s))]
    groups = [(i, g) for i, pf in enumerate(pfs) for g in range(pf.num_row_groups)]
    imgs, ys = [], []
    for i, g in rng.sample(groups, n // 2):
        t = pfs[i].read_row_group(g, columns=["image", "label"])
        for j in rng.sample(range(t.num_rows), 2):
            imgs.append(decode_jpeg(t["image"][j].as_py()["bytes"]))
            ys.append(t["label"][j].as_py())
    return np.stack(imgs), np.array(ys)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--t_start", type=int, default=100, help="DDPM timestep (of 1000); sigma = 1 - t/1000")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float32).to("cuda")
    tr, vae, sf = pipe.transformer, pipe.vae, pipe.vae.config.scaling_factor
    sch = DDIMScheduler.from_config(pipe.scheduler.config)
    sch.set_timesteps(50)
    ts = [t for t in sch.timesteps if t <= args.t_start]
    imgs, ys = real_with_labels(args.n, args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    z0s, zsds = [], []
    for i in range(0, args.n, args.batch):
        x = from_uint8(imgs[i:i + args.batch])
        z0 = vae.encode(x).latent_dist.sample(generator=gen)
        y = torch.tensor(ys[i:i + args.batch], device="cuda")
        noise = torch.randn(z0.shape, generator=gen, device="cuda")
        z = sch.add_noise(z0 * sf, noise, torch.tensor([ts[0]], device="cuda"))
        for t in ts:
            inp = torch.cat([z, z])
            lab = torch.cat([y, torch.full_like(y, 1000)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = tr(inp, timestep=torch.full((len(inp),), t, device="cuda"), class_labels=lab).sample.float()
            e_c, e_u = out[:, :4].chunk(2)
            z = sch.step(e_u + args.cfg * (e_c - e_u), t, z).prev_sample
        z0s.append(z0.cpu().half())
        zsds.append((z / sf).cpu().half())
        print(f"{i + args.batch}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z0=torch.cat(z0s).numpy(), z_sd=torch.cat(zsds).numpy(), y=ys)


if __name__ == "__main__":
    main()
