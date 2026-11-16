"""Color statistics of decoded samples vs real images: does SAD move color / exposure toward real data?

Decodes the first N DiT-XL/2 (DPM-50, cfg 1.5) evaluation latents with D_0 and SAD and compares with N held-out real
ImageNet images (ADM reference, half B): mean saturation (HSV S), mean brightness (V), fraction of clipped pixels
(any channel >= 250 or all channels <= 5), and colorfulness (Hasler & Suesstrunk). Also the Frechet-free per-image
distance of these statistics to the real distribution is summarised by the gap |mean(stat) - mean_real(stat)|.
usage: python analysis/exp_color_stats.py [N]   -> results/color_stats.json
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from eval_metrics import domain_data  # noqa: E402
from refiner import maybe_load_refiner  # noqa: E402
from vae_utils import load_vae, to_uint8  # noqa: E402


def decode(z, ckpt):
    vae = load_vae()
    dec = maybe_load_refiner(vae, ckpt)
    with torch.no_grad():
        return np.concatenate([to_uint8(dec(z[i:i + 25].cuda())) for i in range(0, len(z), 25)])

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def stats(u8):
    x = u8.astype(np.float32)
    hsv = np.stack([np.asarray(Image.fromarray(im).convert("HSV"), np.float32) for im in u8])
    s, v = hsv[..., 1] / 255, hsv[..., 2] / 255
    hi = (x >= 250).any(-1).mean((1, 2))
    lo = (x <= 5).all(-1).mean((1, 2))
    rg, yb = x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    col = np.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2) + 0.3 * np.sqrt(rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2)
    return {"saturation": s.mean((1, 2)), "brightness": v.mean((1, 2)), "clip_high": hi, "clip_low": lo, "colorfulness": col}


fs, _, real, _ = domain_data("imagenet")
z = torch.from_numpy(np.concatenate([np.load(f)["z"] for f in fs])[:N]).float()
S = {"real": stats(real[:N]), "D0": stats(decode(z, "")), "SAD": stats(decode(z, "runs/v3/gen_gan0.05/decoder.pt"))}
out = {}
for k in S["real"]:
    r = float(S["real"][k].mean())
    out[k] = {n: float(S[n][k].mean()) for n in S}
    out[k]["gap_D0"] = abs(out[k]["D0"] - r)
    out[k]["gap_SAD"] = abs(out[k]["SAD"] - r)
for k, v in out.items():
    print(f"{k:13s} real {v['real']:.4f}  D0 {v['D0']:.4f}  SAD {v['SAD']:.4f}  gap D0 {v['gap_D0']:.4f}  gap SAD {v['gap_SAD']:.4f}")
json.dump(out, open("results/color_stats.json", "w"), indent=1)
