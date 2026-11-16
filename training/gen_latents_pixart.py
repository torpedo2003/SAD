"""Sample PixArt-alpha (256px) latents for COCO captions. PixArt-alpha uses the SD-VAE (ft-EMA) decoder.

Saves VAE-space latents (divided by the scaling factor) and the caption indices.
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="json list of strings")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1250)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    caps = json.load(open(a.captions))[a.start:a.start + a.n]
    tr = Transformer2DModel.from_pretrained("pretrained/hf/pixart256", subfolder="transformer", torch_dtype=torch.float16)
    pipe = PixArtAlphaPipeline.from_pretrained("pretrained/hf/pixart512", transformer=tr, variant="fp16",
                                               torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    zs = []
    for i in range(0, len(caps), a.batch):
        lat = pipe(caps[i:i + a.batch], height=256, width=256, num_inference_steps=a.steps, guidance_scale=a.cfg,
                   generator=gen, output_type="latent", use_resolution_binning=False).images
        zs.append((lat.float() / pipe.vae.config.scaling_factor).cpu().half())
        print(f"{i + a.batch}/{len(caps)}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=np.arange(a.start, a.start + len(caps)))


if __name__ == "__main__":
    main()
