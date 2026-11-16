"""Evaluate a (fine-tuned) decoder.

gen sets (decoded generated latents) vs ADM reference batch (10k):
  FID (Inception, pytorch-fid), FD_CLIP (held-out space), FD_DINO (training space, reported only),
  precision / recall (k=3, Inception space).
recon (real75: 10k real images, classes 0-74): rFID vs originals, PSNR.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import fid
from common import CLIP, DINO, FeatNet
from g0_process import load_set
from vae_utils import decode, from_uint8, load_vae, to_uint8


@torch.no_grad()
def feats_all(imgs_u8, nets):
    out = {"inception": fid.features(imgs_u8)}
    for name, net in nets.items():
        fs = [net(from_uint8(imgs_u8[i:i + 250])).cpu().double() for i in range(0, len(imgs_u8), 250)]
        out[name] = torch.cat(fs).numpy()
    return out


def prec_recall(real, fake, k=3):
    r, f = torch.tensor(real).float().cuda(), torch.tensor(fake).float().cuda()

    def radii(x):
        d = torch.cat([torch.cdist(x[i:i + 1000], x) for i in range(0, len(x), 1000)])
        return d.kthvalue(k + 1, dim=1).values

    def covered(a, b, rb):  # fraction of a inside some ball of b
        return float(torch.cat([(torch.cdist(a[i:i + 1000], b) <= rb[None]).any(1)
                                for i in range(0, len(a), 1000)]).float().mean())

    return covered(f, r, radii(r)), covered(r, f, radii(f))


ALL_SETS = ["all_cfg1.5", "all_cfg1.0", "all_cfg4.0", "all_cfg1.5s20", "sit_cfg1.5", "sit_cfg1.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="decoder.pt; empty = original decoder")
    ap.add_argument("--sets", default="", help="default: every complete set among ALL_SETS")
    ap.add_argument("--ref_split", default="B", help="ADM ref split used as reference ('' = all 10k)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vae = load_vae()
    from refiner import maybe_load_refiner
    dec = maybe_load_refiner(vae, args.ckpt)
    nets = {"clip": FeatNet(CLIP), "dino": FeatNet(DINO)}

    os.makedirs("data/proc", exist_ok=True)
    ref_path = "data/proc/ref_adm_feats_all.npz"
    if os.path.exists(ref_path):
        ref = dict(np.load(ref_path))
    else:
        ref = feats_all(np.load("data/VIRTUAL_imagenet256_labeled.npz")["arr_0"], nets)
        np.savez(ref_path, **ref)
    if args.ref_split:  # disjoint from the training target (split A)
        sel = np.load("data/adm_split.npz")[args.ref_split]
        ref = {k: v[sel] for k, v in ref.items()}
    real = np.load("data/real75.npz")
    r75_path = "data/proc/real75_orig_feats_all.npz"
    if os.path.exists(r75_path):
        r75 = dict(np.load(r75_path))
    else:
        r75 = feats_all(real["images"], nets)
        np.savez(r75_path, **r75)

    res = {"ckpt": args.ckpt, "ref": f"adm_{args.ref_split or 'all'}_{len(ref['inception'])}"}
    if os.path.exists(args.out):  # only compute what is missing
        res.update(json.load(open(args.out)))
    sets = args.sets.split(",") if args.sets else [s for s in ALL_SETS if len(glob.glob(f"data/gen/{s}_r*.npz")) == 8]
    for s in sets:
        if s in res:
            continue
        z = torch.from_numpy(load_set(s)).float()
        imgs = np.concatenate([to_uint8(dec(z[i:i + 500])) for i in range(0, len(z), 500)])
        f = feats_all(imgs, nets)
        p, r = prec_recall(ref["inception"], f["inception"])
        res[s] = {"fid": fid.frechet(fid.stats(f["inception"]), fid.stats(ref["inception"])),
                  "fd_clip": fid.frechet(fid.stats(f["clip"]), fid.stats(ref["clip"])),
                  "fd_dino": fid.frechet(fid.stats(f["dino"]), fid.stats(ref["dino"])),
                  "precision": p, "recall": r}
        np.save(f"{os.path.splitext(args.out)[0]}_{s}_samples64.npy", imgs[:64])
        print(s, res[s], flush=True)

        json.dump(res, open(args.out, "w"), indent=1)
    if "recon" in res:
        return
    if os.environ.get("SAD_VAE") in ("vavae", "e2evavae"):  # other tokenizer: encode once with a fixed seed and cache, so all decoders share the latents
        zp = f"data/proc/real75_{os.environ['SAD_VAE']}_z.npy"
        if not os.path.exists(zp):
            torch.manual_seed(0)
            with torch.no_grad():
                zz = [vae.encode(from_uint8(real["images"][i:i + 50])).latent_dist.sample().cpu() for i in range(0, len(real["images"]), 50)]
            np.save(zp, torch.cat(zz).numpy())
        zr = torch.from_numpy(np.load(zp)).float()
    else:
        zr = torch.from_numpy(real["z_sample"]).float()
    rec = np.concatenate([to_uint8(dec(zr[i:i + 500])) for i in range(0, len(zr), 500)])
    f = feats_all(rec, {})
    mse = ((rec.astype(np.float64) - real["images"]) ** 2).mean((1, 2, 3))
    res["recon"] = {"rfid": fid.frechet(fid.stats(f["inception"]), fid.stats(r75["inception"])),
                    "psnr": float(np.mean(10 * np.log10(255 ** 2 / mse)))}
    print("recon", res["recon"], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
