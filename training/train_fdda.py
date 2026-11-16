"""FDDA-style baseline: our re-implementation of the objective of Fang et al., arXiv:2610.17466, written from their public
code. Decoder-only, on fully generated latents, with an EMA-based Frechet loss over Inception-v3, MAE and SigLIP2:
    L = sum_k FD_k / (stop_grad(FD_k) + 0.01),   optionally + lam_anchor * L1(D(E(x)), D_0(E(x))).

Differences from the paper, forced by our setting: the 40k-latent DiT pool instead of 1.28M latents, fewer steps than a full
40k-step epoch, and MAE/SigLIP2 reference statistics from 50k ImageNet training images instead of the full set.
Inception statistics are the ADM reference (mu, sigma), so FID is NOT a held-out metric for this baseline.

usage: python train_fdda.py --out runs/fdda/NAME [--steps 8000] [--lam_anchor 0]
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

import fid
from common import load_gen_train, real_loader
from vae_utils import from_uint8, load_vae

PT = "pretrained/timm"
BETA, EPS = 0.999, 0.01
TIMM = {"mae": ("vit_large_patch16_224.mae", f"{PT}/mae/model.safetensors"),
        "siglip2": ("vit_so400m_patch16_siglip_256.v2_webli", f"{PT}/siglip2/model.safetensors")}


class Extractor(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        arch, path = TIMM[name]
        self.m = timm.create_model(arch, pretrained=True, pretrained_cfg_overlay=dict(file=path), num_classes=0,
                                   dynamic_img_size=True, dynamic_img_pad=True).cuda().eval().requires_grad_(False)
        cfg = self.m.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda())

    def forward(self, y):  # y in [-1, 1]
        x = F.interpolate((y + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        t = self.m.forward_features((x - self.mean) / self.std)
        return (self.m.pool(t) if self.m.attn_pool is not None else t[:, 0]).float()


class IncExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = fid.inception().requires_grad_(False)

    def forward(self, y):
        return self.m((y + 1) / 2)[0].squeeze(-1).squeeze(-1).float()


def sqrt_psd(s):
    w, v = torch.linalg.eigh(s)
    return v @ torch.diag(w.clamp(min=0).sqrt()) @ v.T


class EmaFd:
    """Frechet distance with EMA moments of the generated features; gradients flow only through the current batch."""

    def __init__(self, mu_r, sigma_r):
        self.mu_r, self.sigma_r = mu_r.double().cuda(), sigma_r.double().cuda()
        self.s_r = sqrt_psd(self.sigma_r)
        self.n, self.mu, self.m2 = 0, torch.zeros_like(self.mu_r), torch.zeros_like(self.sigma_r)

    @torch.no_grad()
    def accumulate(self, f):
        f = f.double()
        self.mu += f.sum(0)
        self.m2 += f.T @ f
        self.n += len(f)

    def finalize(self):
        self.mu /= self.n
        self.m2 /= self.n

    def loss(self, f):
        d = f.double()
        mu = BETA * self.mu + (1 - BETA) * d.mean(0)
        m2 = BETA * self.m2 + (1 - BETA) * (d.T @ d) / len(d)
        sig = m2 - mu[None] * mu[:, None]
        m = self.s_r @ sig @ self.s_r
        ev = torch.linalg.eigvalsh(0.5 * (m + m.T)).clamp(min=1e-12)
        fd = (mu - self.mu_r).dot(mu - self.mu_r) + sig.trace() + self.sigma_r.trace() - 2 * ev.sqrt().sum()
        with torch.no_grad():
            self.mu.mul_(BETA).add_(d.mean(0), alpha=1 - BETA)
            self.m2.mul_(BETA).addmm_(d.T, d, alpha=(1 - BETA) / len(d))
        return fd


@torch.no_grad()
def real_stats(nets, n=50000, path="data/proc/fdda_real_stats.npz"):
    if os.path.exists(path):
        return dict(np.load(path))
    feats = {k: [] for k in nets}
    loader, seen = real_loader(128, seed=123), 0
    while seen < n:
        x = from_uint8(next(loader).numpy())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k, net in nets.items():
                feats[k].append(net(x).cpu().double())
        seen += len(x)
    out = {}
    for k, v in feats.items():
        f = torch.cat(v)[:n].numpy()
        out[f"{k}_mu"], out[f"{k}_sigma"] = f.mean(0), np.cov(f, rowvar=False)
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--bs_anchor", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam_anchor", type=float, default=0.0)
    ap.add_argument("--gen_pool", default="cfg1.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    torch.manual_seed(a.seed)

    vae = load_vae().requires_grad_(False)
    dec0 = copy.deepcopy(vae)
    params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    for p in params:
        p.requires_grad_(True)
    vae.enable_gradient_checkpointing()
    vae.decoder.train()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    warm, floor = 500, 0.1
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    nets = {"inception": IncExtractor(), "mae": Extractor("mae"), "siglip2": Extractor("siglip2")}
    stats = real_stats({k: v for k, v in nets.items() if k != "inception"})
    adm = np.load("data/VIRTUAL_imagenet256_labeled.npz")
    fds = {"inception": EmaFd(torch.from_numpy(adm["mu"]), torch.from_numpy(adm["sigma"]))}
    for k in ("mae", "siglip2"):
        fds[k] = EmaFd(torch.from_numpy(stats[f"{k}_mu"]), torch.from_numpy(stats[f"{k}_sigma"]))

    zgen = load_gen_train(a.gen_pool)
    print("latents", len(zgen), flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # EMA moments start from the frozen decoder
        for i in range(0, len(zgen), a.bs):
            y = dec0.decode(zgen[i:i + a.bs].cuda().float()).sample.float().clamp(-1, 1)
            for k, net in nets.items():
                fds[k].accumulate(net(y))
    for f in fds.values():
        f.finalize()
    print("ema_init_done", flush=True)

    loader = real_loader(max(a.bs_anchor, 16), seed=a.seed)
    log, t0 = open(f"{a.out}/log.jsonl", "w"), time.time()
    for step in range(1, a.steps + 1):
        z = zgen[torch.randint(len(zgen), (a.bs,))].cuda().float()
        loss, vals, l_anc = 0.0, {}, torch.zeros(())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = vae.decode(z).sample.float().clamp(-1, 1)
            feats = {k: net(y) for k, net in nets.items()}
        for k, f in feats.items():
            fd = fds[k].loss(f)
            loss = loss + fd / (fd.detach() + EPS)
            vals[k] = fd.item()
        if a.lam_anchor > 0:
            x = from_uint8(next(loader).numpy())[: a.bs_anchor]
            with torch.no_grad():
                za = vae.encode(x).latent_dist.sample()
                t = dec0.decode(za).sample
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_anc = (vae.decode(za).sample.float() - t).abs().mean()
            loss = loss + a.lam_anchor * l_anc
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            rec = {"step": step, "anchor": float(l_anc), **vals, "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    sd = {k: v.cpu() for k, v in vae.state_dict().items() if k.startswith(("decoder.", "post_quant_conv."))}
    torch.save(sd, f"{a.out}/decoder.pt")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
