"""Encode N random real ImageNet images (from the local parquet shards) with the VAE.

Saves uint8 images, labels, posterior means and posterior samples (raw VAE space).
"""

import argparse
import glob

import numpy as np
import pyarrow.parquet as pq
import torch

from vae_utils import decode_jpeg, encode, from_uint8, load_vae

SHARDS = "data/imagenet-1k-256x256/train-{:05d}-of-00040.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tabs = [pq.read_table(glob.glob(SHARDS.format(int(s)))[0]) for s in args.shards.split(",")]
    rows = [(t, i) for t in tabs for i in range(t.num_rows)]
    idx = np.random.default_rng(args.seed).choice(len(rows), args.n, replace=False)
    ims, ys = [], []
    for k in idx:
        t, i = rows[k]
        ims.append(decode_jpeg(t["image"][i].as_py()["bytes"]))
        ys.append(t["label"][i].as_py())
    ims = np.stack(ims)
    print("images", ims.shape, "labels", min(ys), max(ys), flush=True)

    vae = load_vae()
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    zm, zs = [], []
    for i in range(0, len(ims), 500):
        m, s = encode(vae, from_uint8(ims[i:i + 500]), gen=g)
        zm.append(m)
        zs.append(s)
    np.savez(args.out, images=ims, y=np.array(ys), z_mean=torch.cat(zm).numpy(), z_sample=torch.cat(zs).numpy())


if __name__ == "__main__":
    main()
