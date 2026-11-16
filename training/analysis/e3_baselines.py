"""Cheap decoder-side alternatives on the same DiT latents (public decoders and sharpening), to test whether SAD is more than sharpening.

- unsharp mask x + a (x - blur_sigma(x)) and additive Gaussian noise, each tuned for FID against ADM split A
  (the SAD training target), then evaluated on split B like every other method;
- off-the-shelf decoders sharing the SD latent space: sd-vae-ft-mse and the OpenAI Consistency Decoder.
Writes results/e3_baselines.json (gen metrics vs split B on all_cfg1.5 10k, rFID/PSNR on real75).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import fid
from common import CLIP, DINO, FeatNet
from eval_decoder import feats_all, prec_recall
from g0_process import load_set
from vae_utils import decode, load_vae, to_uint8

OUT = "results/e3_baselines.json"


def blur(x, s):
    k = int(2 * round(3 * s) + 1)
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def apply(u8, fn, bs=500):
    out = []
    for i in range(0, len(u8), bs):
        x = torch.from_numpy(u8[i:i + bs]).cuda().permute(0, 3, 1, 2).float()
        out.append(fn(x).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def unsharp(s, a):
    return lambda x: x + a * (x - blur(x, s))


def noise(s):
    def f(x):
        g = torch.Generator(device="cuda").manual_seed(0)
        return x + s * torch.randn(x.shape, generator=g, device="cuda")
    return f


def metrics(imgs, ref, nets, rec=None, real=None):
    f = feats_all(imgs, nets)
    p, r = prec_recall(ref["inception"], f["inception"])
    m = {"fid": fid.frechet(fid.stats(f["inception"]), fid.stats(ref["inception"])),
         "fd_clip": fid.frechet(fid.stats(f["clip"]), fid.stats(ref["clip"])),
         "fd_dino": fid.frechet(fid.stats(f["dino"]), fid.stats(ref["dino"])), "precision": p, "recall": r}
    if rec is not None:
        fr, fo = fid.features(rec), np.load("data/proc/real75_orig_feats_all.npz")["inception"]
        mse = ((rec.astype(np.float64) - real) ** 2).mean((1, 2, 3))
        m["rfid"] = fid.frechet(fid.stats(fr), fid.stats(fo))
        m["psnr"] = float(np.mean(10 * np.log10(255 ** 2 / mse)))
    return m


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    nets = {"clip": FeatNet(CLIP), "dino": FeatNet(DINO)}
    ref = dict(np.load("data/proc/ref_adm_feats_all.npz"))
    sp = np.load("data/adm_split.npz")
    refA = {k: v[sp["A"]] for k, v in ref.items()}
    refB = {k: v[sp["B"]] for k, v in ref.items()}
    vae = load_vae()
    z = torch.from_numpy(load_set("all_cfg1.5")).float()
    real = np.load("data/real75.npz")
    zr = torch.from_numpy(real["z_sample"]).float()
    d0 = np.concatenate([to_uint8(decode(vae, z[i:i + 500])) for i in range(0, len(z), 500)])
    r0 = np.concatenate([to_uint8(decode(vae, zr[i:i + 500])) for i in range(0, len(zr), 500)])

    # ---- tuned post-processing (selection on split A, report on split B)
    for name, grid in (("unsharp", [(s, a) for s in (0.5, 1.0, 1.5, 2.0) for a in (0.15, 0.3, 0.5, 0.8, 1.2)]),
                       ("noise", [(s,) for s in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)])):
        if name in res:
            continue
        sweep = {}
        for prm in grid:
            fn = unsharp(*prm) if name == "unsharp" else noise(*prm)
            fa = fid.features(apply(d0, fn))
            sweep[str(prm)] = fid.frechet(fid.stats(fa), fid.stats(refA["inception"]))
            print(name, prm, sweep[str(prm)], flush=True)
        best = min(sweep, key=sweep.get)
        prm = eval(best)
        fn = unsharp(*prm) if name == "unsharp" else noise(*prm)
        res[name] = {"params": best, "sweep_fid_A": sweep, **metrics(apply(d0, fn), refB, nets, apply(r0, fn), real["images"])}
        print(name, res[name], flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    # ---- off-the-shelf decoders in the same latent space
    from diffusers import AutoencoderKL, ConsistencyDecoderVAE
    for name in ("ft_mse", "consistency"):
        if name in res or os.environ.get("SKIP_DEC"):
            continue
        if name == "ft_mse":
            dv = AutoencoderKL.from_pretrained("pretrained/hf/sd-vae-ft-mse").cuda().eval()
            dec = lambda zz: dv.decode(zz.cuda()).sample.clamp(-1, 1)
            bs = 50
        else:
            dv = ConsistencyDecoderVAE.from_pretrained("pretrained/hf/consistency-decoder", variant="fp16", torch_dtype=torch.float16).cuda().eval()
            g = torch.Generator(device="cuda").manual_seed(0)
            dec = lambda zz: dv.decode(zz.cuda().half(), generator=g).sample.float().clamp(-1, 1)
            bs = 25
        with torch.no_grad():
            gi = np.concatenate([to_uint8(dec(z[i:i + bs])) for i in range(0, len(z), bs)])
            ri = np.concatenate([to_uint8(dec(zr[i:i + bs])) for i in range(0, len(zr), bs)])
        res[name] = metrics(gi, refB, nets, ri, real["images"])
        np.save(f"results/e3_{name}_samples64.npy", gi[:64])
        print(name, res[name], flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del dv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
