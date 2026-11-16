"""SiT-XL/1 trained with REPA on REPA-E's frozen end-to-end VAE (E2E-VAVAE f16d32; checkpoint REPA-E/sit-ldm-e2e-vavae, 4M
steps). Default sampling is the REPA-E release recipe for this model (FID 1.12): Euler-Maruyama SDE, 250 steps, cfg 2.4 on
the guidance interval [0, 0.73], class-balanced labels (50 per class over the 50k set; --shard picks a 1/8 slice).
--ode uses the cheaper Heun ODE (training pools) with random labels. Latents are saved in raw VAE space."""
import argparse
import os
import sys

import numpy as np
import torch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/REPA-E")
sys.path.insert(0, R)
from models.sit import SiT_models  # noqa: E402
from samplers import euler_maruyama_sampler, euler_sampler  # noqa: E402

P = "pretrained/repae"


def load_model():
    ema = f"{P}/sit_ldm_e2e_vavae_ema.pt"
    if not os.path.exists(ema):  # the release bundles optimizer state (11 GB); keep the EMA weights only
        sd = torch.load(f"{P}/checkpoints/4000000.pt", map_location="cpu", weights_only=False)["ema"]
        torch.save(sd, ema + ".tmp"); os.replace(ema + ".tmp", ema)
    m = SiT_models["SiT-XL/1"](input_size=16, in_channels=32, num_classes=1000, class_dropout_prob=0.1, z_dims=[768],
                               encoder_depth=8, fused_attn=True, qk_norm=False)
    m.load_state_dict(torch.load(ema, map_location="cpu"))
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6250)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=-1, help="class-balanced 50k set: slice index (0-7)")
    ap.add_argument("--cfg", type=float, default=2.4)
    ap.add_argument("--g_high", type=float, default=0.73)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--ode", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    model = load_model()
    st = torch.load(f"{P}/e2e-vavae-400k-latents-stats.pt", map_location="cuda")
    torch.manual_seed(a.seed)
    if a.shard >= 0:
        g = np.random.default_rng(0)
        labels = g.permutation(np.repeat(np.arange(1000), 50))[a.shard * 6250:(a.shard + 1) * 6250][:a.n]
    else:
        labels = np.random.default_rng(a.seed).integers(0, 1000, size=a.n)
    zs = []
    for i in range(0, len(labels), a.batch):
        y = torch.from_numpy(labels[i:i + a.batch]).cuda()
        x = torch.randn(len(y), 32, 16, 16, device="cuda")
        fn = euler_sampler if a.ode else euler_maruyama_sampler
        s = fn(model=model, latents=x, y=y, num_steps=a.steps, heun=a.ode, cfg_scale=a.cfg, guidance_low=0.0,
               guidance_high=a.g_high, path_type="linear").float()
        s = s / st["latents_scale"] + st["latents_bias"]  # REPA-E denormalize_latents
        zs.append(s.cpu().half())
        print(f"{i + len(y)}/{len(labels)}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=labels)


if __name__ == "__main__":
    main()
