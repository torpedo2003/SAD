"""Standard-protocol eval: decode a 50k latent set and compute FID-50K against the ADM ImageNet-256
reference statistics (mu/sigma in VIRTUAL_imagenet256_labeled.npz, the numbers papers report against),
plus precision/recall (k=3) against the 10k ADM reference images and FD_CLIP (held-out space).

Features use pytorch-fid's port of the TF Inception; small offsets vs. the TF evaluator are expected,
so compare methods only within this script.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import fid
from common import CLIP, FeatNet
from eval_decoder import prec_recall
from vae_utils import decode, from_uint8, load_vae, to_uint8
from refiner import maybe_load_refiner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", required=True, help="data/gen/{set}_r*.npz (8 shards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_split_b", action="store_true", help="P/R and FD_CLIP against ADM split B only, disjoint from the training target")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if args.set in res:
        return
    vae = load_vae()
    dec = maybe_load_refiner(vae, args.ckpt)
    clip = FeatNet(CLIP)
    ref = np.load("data/VIRTUAL_imagenet256_labeled.npz")
    ref10k = dict(np.load("data/proc/ref_adm_feats_all.npz"))
    if args.ref_split_b:
        _b = np.load("data/adm_split.npz")["B"]
        ref10k = {k: v[_b] for k, v in ref10k.items()}

    fs = sorted(glob.glob(f"data/gen/{args.set}_r*.npz"))
    assert len(fs) == 8 or os.environ.get("SAD_ALLOW_PARTIAL"), fs
    f_inc, f_clip = [], []
    for f in fs:
        z = torch.from_numpy(np.load(f)["z"]).float()
        for i in range(0, len(z), 500):
            imgs = to_uint8(dec(z[i:i + 500]))
            f_inc.append(fid.features(imgs))
            with torch.no_grad():
                f_clip.append(torch.cat([clip(from_uint8(imgs[j:j + 250])).cpu().double()
                                         for j in range(0, len(imgs), 250)]).numpy())
    f_inc, f_clip = np.concatenate(f_inc), np.concatenate(f_clip)
    p, r = prec_recall(ref10k["inception"], f_inc)
    res[args.set] = {"n": len(f_inc),
                     "fid50k": fid.frechet(fid.stats(f_inc), (ref["mu"], ref["sigma"])),
                     "fid_vs10k": fid.frechet(fid.stats(f_inc), fid.stats(ref10k["inception"])),
                     "fd_clip": fid.frechet(fid.stats(f_clip), fid.stats(ref10k["clip"])),
                     "precision": p, "recall": r}
    print(args.set, res[args.set], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
