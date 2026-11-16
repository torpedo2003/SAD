"""DiT-XL/2 latents with Discriminator Guidance (Kim et al., ICML 2023), using the official DiT code and checkpoint (same
as gen_latents_dit_official.py) and the released DG discriminator (alsdudrla10/DG_imagenet, discriminator_7.pt on top
of their 32x32x4 ADM classifier). The DG paper recipe is --lt 1.25 --st 3.0 --branch 200 --dg 1.0 --time_min 100
(their FID 1.83); --lt 1.5 --st 1.5 is DG under the plain cfg 1.5 of our DiT baseline; --dg 0 disables the guidance.
Latents are saved in raw VAE space (divided by 0.18215)."""
import argparse
import os
import sys

import numpy as np
import torch

L = "third_party/DiT"
DG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/DG_imagenet")
sys.path.insert(0, L)
from diffusion import create_diffusion  # noqa: E402
from models import DiT_models  # noqa: E402

sys.path.insert(0, DG)
import classifier_lib  # noqa: E402

CK = "pretrained/dit/DiT-XL-2-256x256.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lt", type=float, default=1.25, help="cfg scale for t > branch")
    ap.add_argument("--st", type=float, default=3.0, help="cfg scale for t <= branch")
    ap.add_argument("--branch", type=int, default=200)
    ap.add_argument("--dg", type=float, default=1.0)
    ap.add_argument("--time_min", type=int, default=100)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast for the DiT forward only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).cuda()
    model.load_state_dict(torch.load(CK, map_location="cpu", weights_only=False))
    model.eval()
    diffusion = create_diffusion("250")
    pm = f"{DG}/pretrained_models"
    disc = classifier_lib.get_adm_discriminator(f"{pm}/ADM_classifier/32x32_classifier.pt",
                                                f"{pm}/discriminator/discriminator_7.pt", True, 32, device="cuda", enable_grad=True)

    def fwd(x, t, y):  # DG_imagenet models.forward_with_dg: two cfg scales split at `branch`
        half = x[: len(x) // 2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
            out = model.forward(torch.cat([half, half]), t, y).float()
        eps, rest = out[:, :3], out[:, 3:]
        c, u = eps.chunk(2)
        s = a.lt if t[0] > a.branch else a.st
        h = u + s * (c - u)
        return torch.cat([torch.cat([h, h]), rest], 1)

    def cond_fn(x, t, y=None, **kw):  # DG_imagenet sample.py
        if a.dg == 0 or t[0] <= a.time_min:
            return torch.zeros_like(x)
        n = len(x) // 2
        with torch.enable_grad():
            xi = x[:n].detach().requires_grad_(True)
            pr = disc(xi, t[:n] / 999, condition=y[:n]).clip(1e-5, 1 - 1e-5)
            g = torch.autograd.grad((pr.log() - (1 - pr).log()).sum(), xi)[0] * a.dg
        return torch.cat([g, g])

    zs, ys, done = [], [], 0
    while done < a.n:
        n = min(a.batch, a.n - done)
        z = torch.randn(n, 4, 32, 32, device="cuda")
        y = torch.randint(0, 1000, (n,), device="cuda")
        zc, yc = torch.cat([z, z]), torch.cat([y, torch.full_like(y, 1000)])
        s = diffusion.p_sample_loop(fwd, zc.shape, zc, clip_denoised=False, model_kwargs=dict(y=yc), progress=False,
                                    device="cuda", cond_fn=cond_fn)
        zs.append((s.chunk(2)[0] / 0.18215).cpu().float()); ys.append(y.cpu())
        done += n
        print(f"{done}/{a.n}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=torch.cat(zs).numpy(), y=torch.cat(ys).numpy())


if __name__ == "__main__":
    main()
