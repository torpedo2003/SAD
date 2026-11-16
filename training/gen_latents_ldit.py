"""LightningDiT-XL/1 (800 epochs) latents in raw VA-VAE space. Sampling follows the official FID=1.35 configuration:
Euler ODE, 250 steps, CFG 6.7 with guidance interval start 0.125, timestep shift 0.3, class labels uniform."""
import argparse
import os
import sys

import numpy as np
import torch

L = "third_party/LightningDiT"
sys.path.insert(0, L)
from models.lightningdit import LightningDiT_models  # noqa: E402
from transport import Sampler, create_transport  # noqa: E402

P = "pretrained/lightningdit"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = LightningDiT_models["LightningDiT-XL/1"](input_size=16, num_classes=1000, use_qknorm=False, use_swiglu=True,
                                                     use_rope=True, use_rmsnorm=True, wo_shift=False, in_channels=32)
    model.load_state_dict(torch.load(f"{P}/lightningdit-xl-imagenet256-800ep.pt", map_location="cpu", weights_only=False)["ema"])
    model = model.cuda().eval()
    transport = create_transport("Linear", "velocity", None, None, None, use_cosine_loss=True, use_lognorm=True)
    sample_fn = Sampler(transport).sample_ode(sampling_method="euler", num_steps=250, atol=1e-6, rtol=1e-3, reverse=False, timestep_shift=0.3)
    st = torch.load(f"{P}/latents_stats.pt", map_location="cuda", weights_only=True)
    labels, zs = np.random.default_rng(a.seed).integers(0, 1000, size=a.n), []
    for i in range(0, a.n, a.batch):
        y = torch.from_numpy(labels[i:i + a.batch]).cuda()
        n = len(y)
        z = torch.randn(n, 32, 16, 16, device="cuda")
        z, ys = torch.cat([z, z]), torch.cat([y, torch.full_like(y, 1000)])
        s = sample_fn(z, model.forward_with_cfg, y=ys, cfg_scale=6.7, cfg_interval=True, cfg_interval_start=0.125)[-1]
        s = s.chunk(2)[0] * st["std"] + st["mean"]
        zs.append(s.cpu().half())
        print(f"{i + n}/{a.n}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
