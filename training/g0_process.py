"""G0 per-set processing (sharded over GPUs).

For a latent set z: x = D(z); P(z) = E_mean(x); x2 = D(P(z)).
Saves P(z), Inception features of x and x2, and per-sample pixel MSE between x and x2.
"""

import argparse
import glob

import numpy as np
import torch

import fid
from vae_utils import decode, encode, load_vae, to_uint8


def load_set(name):
    if name.startswith("real_"):
        d = np.load("data/real75.npz")
        return d["z_" + name[5:]]
    fs = sorted(glob.glob(f"data/gen/{name}_r*.npz"))
    assert len(fs) == 8, fs
    return np.concatenate([np.load(f)["z"] for f in fs]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()

    z = load_set(args.set)
    chunk = np.array_split(np.arange(len(z)), args.world)[args.rank]
    z = torch.from_numpy(z[chunk]).float()
    vae = load_vae()
    pz, f1, f2, mse = [], [], [], []
    for i in range(0, len(z), 250):
        x = decode(vae, z[i:i + 250])
        m, _ = encode(vae, x)
        x2 = decode(vae, m)
        pz.append(m)
        mse.append(((x - x2) ** 2).mean((1, 2, 3)).cpu().numpy())
        f1.append(fid.features(to_uint8(x)))
        f2.append(fid.features(to_uint8(x2)))
    np.savez(f"data/proc/{args.set}_r{args.rank}.npz", idx=chunk, Pz=torch.cat(pz).numpy(),
             f_D=np.concatenate(f1), f_DP=np.concatenate(f2), mse_img=np.concatenate(mse))
    print("done", args.set, args.rank, flush=True)


if __name__ == "__main__":
    main()
