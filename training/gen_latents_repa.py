"""Sample SiT-XL/2 + REPA latents (official 4M-iteration checkpoint) with the settings of the REPA paper
(SDE Euler-Maruyama, 250 steps, cfg 1.8, guidance interval [0, 0.7]; FID 1.42 reported). REPA uses SD-VAE ft-EMA.

Saves VAE-space latents (divided by 0.18215) and class labels, like gen_latents.py.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/REPA"))
from models.sit import SiT_models  # noqa: E402
from samplers import euler_maruyama_sampler, euler_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1250)
    ap.add_argument("--cfg", type=float, default=1.8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--g_high", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--ckpt", default="pretrained/repa_sit_xl2_last.pt")
    ap.add_argument("--ode", action="store_true", help="Heun ODE sampler instead of the SDE (cheaper; for training pools)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000, use_cfg=True, z_dims=[768], encoder_depth=8,
                                   fused_attn=False, qk_norm=False).cuda()
    sd = torch.load(a.ckpt, map_location="cuda")
    model.load_state_dict(sd["ema"] if "ema" in sd else sd)
    model.eval()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    labels = rng.integers(0, 1000, size=a.n)
    zs = []
    for i in range(0, a.n, a.batch):
        y = torch.tensor(labels[i:i + a.batch], device="cuda")
        x = torch.randn(len(y), 4, 32, 32, device="cuda")
        with torch.no_grad():
            fn = euler_sampler if a.ode else euler_maruyama_sampler
            kw = dict(heun=True) if a.ode else {}
            s = fn(model=model, latents=x, y=y, num_steps=a.steps, cfg_scale=a.cfg, guidance_low=0.0,
                   guidance_high=a.g_high, path_type="linear", **kw).float()
        zs.append((s / 0.18215).cpu().half())
        print(f"{i + a.batch}/{a.n}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
