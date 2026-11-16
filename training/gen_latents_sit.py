"""Sample SiT-XL/2-256 latents (same VAE as DiT: sd-vae-ft-ema) -> same npz format as gen_latents.py.

SiT convention (linear interpolant, velocity prediction): x_t = t * x_data + (1 - t) * noise,
t goes 0 (noise) -> 1 (data), model(x_t, t, y) = dx_t/dt. Heun ODE solver on a uniform grid.
CFG is applied to all 4 latent channels.
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SiT"))
from models import SiT_XL_2  # noqa: E402

CKPT = "pretrained/SiT-XL-2-256x256.safetensors"
SCALE = 0.18215


@torch.no_grad()
def velocity(model, x, t, y, cfg):
    if cfg > 1:
        inp, ts, lab = torch.cat([x, x]), t.repeat(2), torch.cat([y, torch.full_like(y, 1000)])
    else:
        inp, ts, lab = x, t, y
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(inp, ts, lab).float()
    if cfg > 1:
        vc, vu = v.chunk(2)
        v = vu + cfg * (vc - vu)
    return v


@torch.no_grad()
def sample(model, labels, cfg, steps, generator):
    y = torch.tensor(labels, device="cuda")
    x = torch.randn((len(labels), 4, 32, 32), generator=generator, device="cuda")
    ts = torch.linspace(0, 1, steps + 1, device="cuda")
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        d0 = velocity(model, x, t0.expand(len(x)), y, cfg)
        xe = x + (t1 - t0) * d0
        if i < steps - 1:  # Heun correction (Euler on the last step)
            d1 = velocity(model, xe, t1.expand(len(x)), y, cfg)
            x = x + (t1 - t0) * (d0 + d1) / 2
        else:
            x = xe
    return x / SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=False).cuda().eval()
    sd = load_file(CKPT)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    print(model.load_state_dict(sd, strict=True), flush=True)
    rng = np.random.default_rng(args.seed)
    labels = rng.integers(0, 1000, size=args.n)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    zs = []
    for i in range(0, args.n, args.batch):
        zs.append(sample(model, labels[i:i + args.batch].tolist(), args.cfg, args.steps, gen).cpu().half())
        print(f"{i + args.batch}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
