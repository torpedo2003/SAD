"""Decode the eight showcase DiT-XL/2 latents with the original decoder (D_0) and with SAD.

    python demo.py                      # -> assets/demo.jpg, the D_0 / SAD / difference figure shown in the README

The latents (DiT-XL/2, 50-step DPM-Solver++, guidance 1.5) are stored in assets/showcase_latents.npz, so no generator
and no sampling are needed: only the decoder weights are swapped.
"""
import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

from sad import load_sad_decoder


def heat(d):
    anchors = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21),
                        (250, 193, 39), (252, 255, 164)], np.float32)
    x = np.clip(d / np.percentile(d, 99.5), 0, 1)
    t = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(x, t, anchors[:, c]) for c in range(3)], -1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dit-xl2-256-dpm50/sad-gan", help="SAD checkpoint name (see sad/hub.py)")
    ap.add_argument("--local_path", default=None, help="load the checkpoint from a local .safetensors file")
    ap.add_argument("--out", default="assets/demo.jpg")
    a = ap.parse_args()

    z = torch.from_numpy(np.load("assets/showcase_latents.npz")["z"]).cuda()
    vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae").cuda().eval()

    @torch.no_grad()
    def decode():
        x = vae.decode(z).sample.clamp(-1, 1)
        return ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    d0 = decode()                                              # original decoder D_0
    load_sad_decoder(vae, a.ckpt, local_path=a.local_path)     # swap in the SAD decoder
    sad = decode()

    s, g = 256, 4
    canvas = Image.new("RGB", (len(z) * (s + g), 3 * (s + g)), "white")
    for k in range(len(z)):
        diff = heat(np.abs(sad[k].astype(np.float32) - d0[k]).mean(-1))
        for r, im in enumerate([d0[k], sad[k], diff]):
            canvas.paste(Image.fromarray(im), (k * (s + g), r * (s + g)))
    canvas.save(a.out, quality=92)
    print(f"saved {a.out}: rows = D_0, SAD, |SAD - D_0|")


if __name__ == "__main__":
    main()
