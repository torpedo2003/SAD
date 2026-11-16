"""FD_CLIP (held-out ADM split B) and a pytorch FID-50K sanity check for a released or externally sampled image set.
Same feature networks and reference as fid50k.py --ref_split_b; input is uint8 images instead of latents.
usage: python fdclip_images.py NAME OUT.json FILE.npz [FILE.npz ...]   (key 'arr_0' or 'images')
"""

import json
import os
import sys

import numpy as np
import torch

import fid
from common import CLIP, FeatNet
from eval_decoder import prec_recall
from vae_utils import from_uint8


def main():
    name, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    clip = FeatNet(CLIP)
    ref = np.load("data/VIRTUAL_imagenet256_labeled.npz")
    ref10k = dict(np.load("data/proc/ref_adm_feats_all.npz"))
    _b = np.load("data/adm_split.npz")["B"]
    ref10k = {k: v[_b] for k, v in ref10k.items()}
    f_inc, f_clip = [], []
    for f in files:
        d = np.load(f, mmap_mode="r")
        x = d["arr_0"] if "arr_0" in d.files else d["images"]
        for i in range(0, len(x), 500):
            imgs = np.ascontiguousarray(x[i:i + 500])
            f_inc.append(fid.features(imgs))
            with torch.no_grad():
                f_clip.append(torch.cat([clip(from_uint8(imgs[j:j + 250])).cpu().double()
                                         for j in range(0, len(imgs), 250)]).numpy())
            print(f, i, flush=True)
    f_inc, f_clip = np.concatenate(f_inc), np.concatenate(f_clip)
    p, r = prec_recall(ref10k["inception"], f_inc)
    res = json.load(open(out)) if os.path.exists(out) else {}
    res[name] = {"n": len(f_inc), "fid50k": fid.frechet(fid.stats(f_inc), (ref["mu"], ref["sigma"])),
                 "fd_clip": fid.frechet(fid.stats(f_clip), fid.stats(ref10k["clip"])), "precision": p, "recall": r}
    print(name, res[name], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
