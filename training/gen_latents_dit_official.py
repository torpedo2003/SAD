"""DiT-XL/2 latents from the official facebookresearch/DiT code and checkpoint (250 steps, cfg 1.5, fp32), without any
diffusers component, to find where the FID 2.27 of the DiT paper differs from our diffusers-based reproduction.
Latents are saved in raw VAE space (divided by 0.18215)."""
import argparse
import os
import sys

import numpy as np
import torch

L = "third_party/DiT"
sys.path.insert(0, L)
from diffusion import create_diffusion  # noqa: E402
from models import DiT_models  # noqa: E402

CK = "pretrained/dit/DiT-XL-2-256x256.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).cuda()
    model.load_state_dict(torch.load(CK, map_location="cpu", weights_only=False))
    model.eval()
    diffusion = create_diffusion("250")
    zs, ys, done = [], [], 0
    while done < a.n:
        n = min(a.batch, a.n - done)
        z = torch.randn(n, 4, 32, 32, device="cuda")
        y = torch.randint(0, 1000, (n,), device="cuda")
        zc, yc = torch.cat([z, z]), torch.cat([y, torch.full_like(y, 1000)])
        s = diffusion.p_sample_loop(model.forward_with_cfg, zc.shape, zc, clip_denoised=False,
                                    model_kwargs=dict(y=yc, cfg_scale=1.5), progress=False, device="cuda")
        zs.append((s.chunk(2)[0] / 0.18215).cpu().float()); ys.append(y.cpu())
        done += n
        print(f"{done}/{a.n}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=torch.cat(ys).numpy())


if __name__ == "__main__":
    main()
