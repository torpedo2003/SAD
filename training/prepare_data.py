"""Prepare the real-image data used for training and evaluation.

1. data/adm_split.npz   random split of the 10k ADM ImageNet-256 reference images:
                        A (5k) is the only real-image set used by GSA, B (5k) is held out for evaluation.
2. data/real75.npz      10k ImageNet training images (and their VAE latents) for rFID / PSNR.

Expects data/VIRTUAL_imagenet256_labeled.npz and the ImageNet-256 parquet shards in data/imagenet-1k-256x256/
(see training/README.md).

usage: python prepare_data.py [--skip_real75]
"""
import argparse
import os
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_real75", action="store_true")
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    assert os.path.exists("data/VIRTUAL_imagenet256_labeled.npz"), "download the ADM reference batch first"
    n = len(np.load("data/VIRTUAL_imagenet256_labeled.npz")["arr_0"])
    p = np.random.default_rng(0).permutation(n)
    np.savez("data/adm_split.npz", A=np.sort(p[: n // 2]), B=np.sort(p[n // 2:]))
    print(f"data/adm_split.npz: A={n // 2}, B={n - n // 2}")
    if not a.skip_real75:
        subprocess.run([sys.executable, "encode_real.py", "--shards", "0,1,2", "--n", "10000", "--seed", "0",
                        "--out", "data/real75.npz"], check=True)


if __name__ == "__main__":
    main()
