"""SiT-XL/2-256 latents with the official SiT recipe that reports FID 2.06 (Tab. 1, FD_CLIP only):
SDE sampler from the official `transport` package (Euler-Maruyama, diffusion form "sigma", last step "Mean" of size 0.04,
250 steps), guidance 1.5 through model.forward_with_cfg (three latent channels, as in the official sample_ddp.py), random labels.
Output: same npz format as gen_latents_sit.py (z already divided by the scaling factor, y labels).
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SiT"))
from models import SiT_XL_2  # noqa: E402
from transport import Sampler, create_transport  # noqa: E402

CKPT = "pretrained/SiT-XL-2-256x256.safetensors"
SCALE = 0.18215


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=False).cuda().eval()
    sd = {k.removeprefix("module."): v for k, v in load_file(CKPT).items()}
    print(model.load_state_dict(sd, strict=True), flush=True)
    sample_fn = Sampler(create_transport("Linear", "velocity")).sample_sde(
        sampling_method="Euler", diffusion_form="sigma", diffusion_norm=1.0, last_step="Mean", last_step_size=0.04,
        num_steps=args.steps)
    torch.manual_seed(args.seed)
    labels = np.random.default_rng(args.seed).integers(0, 1000, size=args.n)
    zs = []
    with torch.no_grad():
        for i in range(0, args.n, args.batch):
            y = torch.tensor(labels[i:i + args.batch], device="cuda")
            z = torch.randn(len(y), 4, 32, 32, device="cuda")
            z = torch.cat([z, z])
            y = torch.cat([y, torch.full_like(y, 1000)])
            x = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=args.cfg)[-1]
            zs.append((x.chunk(2)[0] / SCALE).cpu().half())
            print(f"{i + args.batch}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
