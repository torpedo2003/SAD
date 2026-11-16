"""Sample DiT-XL/2 latents (before VAE decoding) with classifier-free guidance.

Saves VAE-space latents (already divided by the scaling factor) so that
`vae.decode(z)` gives the image, plus the class labels.
"""

import argparse
import os

import numpy as np
import torch
from diffusers import DDPMScheduler, DiTPipeline, DPMSolverMultistepScheduler


@torch.no_grad()
def sample(pipe, labels, cfg, steps, generator, gi=(0.0, float("inf")), cfg_ch=None, adm_ts=False, amp=True):
    """gi: guidance interval (Kynkaanniemi et al. 2024) in noise level sigma = sqrt((1-ab)/ab); cfg=1 outside."""
    tr, sch = pipe.transformer, pipe.scheduler
    c = tr.config.in_channels
    size = tr.config.sample_size
    n = len(labels)
    x = torch.randn((n, c, size, size), generator=generator, device="cuda", dtype=torch.float32) * sch.init_noise_sigma
    cond = torch.tensor(labels, device="cuda")
    null = torch.full_like(cond, 1000)
    if adm_ts:  # ADM/DiT respacing: evenly spaced, including t=999 and t=0 (diffusers "leading" starts at 996)
        sch.set_timesteps(timesteps=[round(i * 999 / (steps - 1)) for i in range(steps)][::-1])
    else:
        sch.set_timesteps(steps)
    for t in sch.timesteps:
        inp = torch.cat([x, x]) if cfg > 1 else x
        inp = sch.scale_model_input(inp, t)
        lab = torch.cat([cond, null]) if cfg > 1 else cond
        ts = torch.full((inp.shape[0],), t, device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = tr(inp, timestep=ts, class_labels=lab).sample.float()
        eps, var = out[:, :c], out[:, c:]
        if cfg > 1:
            ab = sch.alphas_cumprod[int(t)].item()
            w = cfg if gi[0] < ((1 - ab) / ab) ** 0.5 <= gi[1] else 1.0
            e_c, e_u = eps.chunk(2)
            g = e_u + w * (e_c - e_u)
            if cfg_ch:  # official DiT: guidance on the first `cfg_ch` channels only (forward_with_cfg)
                g = torch.cat([g[:, :cfg_ch], e_c[:, cfg_ch:]], 1)
            eps, var = g, var.chunk(2)[0]
        if isinstance(sch, DDPMScheduler):  # learned-variance ancestral sampling (official DiT setting)
            x = sch.step(torch.cat([eps, var], 1), t, x, generator=generator).prev_sample
        else:
            x = sch.step(eps, t, x).prev_sample
    return x / pipe.vae.config.scaling_factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--label_max", type=int, default=1000, help="sample labels in [0, label_max)")
    ap.add_argument("--gi", default="", help="guidance interval 'lo,hi' in sigma (empty = everywhere)")
    ap.add_argument("--sampler", choices=["dpm", "ddpm"], default="dpm")
    ap.add_argument("--model", default="facebook/DiT-XL-2-256", help="or facebook/DiT-XL-2-512")
    ap.add_argument("--cfg_channels", type=int, default=0, help="3 = official DiT guidance (first 3 channels only)")
    ap.add_argument("--adm_ts", action="store_true", help="ADM/DiT timestep respacing (DDPM only)")
    ap.add_argument("--no_amp", action="store_true", help="run the transformer in fp32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float32).to("cuda")
    if args.sampler == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, variance_type="learned_range")
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    rng = np.random.default_rng(args.seed)
    labels = rng.integers(0, args.label_max, size=args.n)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    zs = []
    for i in range(0, args.n, args.batch):
        zs.append(sample(pipe, labels[i:i + args.batch].tolist(), args.cfg, args.steps, gen,
                         tuple(map(float, args.gi.split(","))) if args.gi else (0.0, float("inf")),
                         args.cfg_channels, args.adm_ts, not args.no_amp).cpu().half())
        print(f"{i + args.batch}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
