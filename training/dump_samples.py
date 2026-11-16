"""Decode a 50k latent set and write an ADM-style sample batch for the official ADM evaluator.

    python dump_samples.py --set dit50k_cfg1.5 --name sad_gan --ckpt runs/sad_gan/decoder.pt
    -> data/samples/dit50k_cfg1.5__sad_gan.npz  (key arr_0, uint8 [N,256,256,3])

Leave --ckpt empty for the original decoder D_0. Then run, from openai/guided-diffusion/evaluations:
    python evaluator.py VIRTUAL_imagenet256_labeled.npz data/samples/dit50k_cfg1.5__sad_gan.npz
"""
import argparse
import glob
import os

import numpy as np
import torch

from refiner import maybe_load_refiner
from vae_utils import load_vae, to_uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="data/gen/{set}_r*.npz")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    fs = sorted(glob.glob(f"data/gen/{a.set}_r*.npz"))
    assert fs, a.set
    dec = maybe_load_refiner(load_vae(), a.ckpt)
    out = []
    with torch.no_grad():
        for f in fs:
            z = torch.from_numpy(np.load(f)["z"]).float()
            for i in range(0, len(z), 250):
                out.append(to_uint8(dec(z[i:i + 250])))
    os.makedirs("data/samples", exist_ok=True)
    path = f"data/samples/{a.set}__{a.name}.npz"
    np.savez(path, arr_0=np.concatenate(out))
    print("saved", path)


if __name__ == "__main__":
    main()
