"""SAD: fine-tune only the VAE decoder of a frozen latent generator (GSA + EAP).

loss = L1(D_phi(E(x)), D_0(E(x)))                      # EAP anchor: unchanged on the encoder manifold (x: mirror images)
     + lam * L_dist(D_phi(z_src), real target)          # z_src = generated latents (mode=gen)
                                                        #       or encoder latents   (mode=enc, the GSA-on-E(x) control)
L_dist in DINOv2-S space: fd (queue-based Frechet), mmd, or gan (multi-layer DINOv2 token discriminator, hinge).
Real target (--target adm_a): 5k clean ADM-reference images (split A); evaluation uses the disjoint split B.
The mirror images are JPEG q75 re-encodes, so they are NOT used as the distribution target.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DINO, FeatNet, load_gen_train, load_gen_train_labels, real_loader
from refiner import Refiner
from vae_utils import from_uint8, load_vae


def sqrtm_psd(c):
    w, v = torch.linalg.eigh(c)
    return (v * w.clamp_min(0).sqrt()) @ v.T


def frechet_loss(f, mu_r, c_r, sqrt_c_r):
    mu = f.mean(0)
    fc = f - mu
    c = fc.T @ fc / (len(f) - 1)
    m = sqrt_c_r @ c @ sqrt_c_r
    tr_sqrt = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(1e-10).sqrt().sum()
    return ((mu - mu_r) ** 2).sum() + torch.trace(c) + torch.trace(c_r) - 2 * tr_sqrt


def mmd_loss(f, r):
    """Unbiased RBF-MMD^2 with multiple bandwidths (relative to median distance of r)."""
    x = torch.cat([f, r])
    d = torch.cdist(x, x) ** 2
    med = d[len(f):, len(f):].detach().median()
    k = sum(torch.exp(-d / (s * med)) for s in (0.25, 0.5, 1, 2, 4))
    n, m = len(f), len(r)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return ((kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
            - 2 * kxy.mean())


def rand_crop_zoom(x, size, gen):
    """Same-size random crop (size x size) per sample, resized back to the input resolution (2x zoom for 128/256)."""
    n, _, h, w = x.shape
    ys = torch.randint(0, h - size + 1, (n,), device=x.device, generator=gen)
    xs = torch.randint(0, w - size + 1, (n,), device=x.device, generator=gen)
    out = torch.stack([x[i, :, ys[i]:ys[i] + size, xs[i]:xs[i] + size] for i in range(n)])
    return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def diff_augment(x, gen):
    """Same random flip + translation (+-1/8) applied per-sample; used on real and fake alike."""
    n = len(x)
    flip = torch.rand(n, 1, 1, 1, device=x.device, generator=gen) < 0.5
    x = torch.where(flip, x.flip(3), x)
    s = x.shape[-1] // 8
    pad = F.pad(x, (s, s, s, s), mode="reflect")
    dx = torch.randint(0, 2 * s + 1, (n,), device=x.device, generator=gen)
    dy = torch.randint(0, 2 * s + 1, (n,), device=x.device, generator=gen)
    idx = torch.arange(x.shape[-1], device=x.device)
    rows = (dy[:, None] + idx[None]).view(n, 1, -1, 1).expand(-1, x.shape[1], -1, pad.shape[-1])
    pad = pad.gather(2, rows)
    cols = (dx[:, None] + idx[None]).view(n, 1, 1, -1).expand(-1, x.shape[1], x.shape[-1], -1)
    return pad.gather(3, cols)


class TokenDisc(nn.Module):
    """Per-token discriminator heads on frozen DINOv2 blocks (projected-GAN / ADD style)."""

    def __init__(self, psi, layers=(2, 5, 8, 11), dim=384, n_classes=0):
        super().__init__()
        self.psi, self.layers = psi, layers
        self.heads = nn.ModuleList(nn.Sequential(nn.Conv1d(dim, 256, 1), nn.LeakyReLU(0.2),
                                                 nn.Conv1d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
                                                 nn.Conv1d(256, 1, 1)) for _ in layers)
        # class-conditional projection discriminator (Miyato & Koyama 2018); off by default
        self.emb = nn.ModuleList(nn.Embedding(n_classes, 256) for _ in layers) if n_classes else None

    def forward(self, x, y=None):
        x = (x + 1) / 2
        x = F.interpolate(x.float(), size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.psi.mean) / self.psi.std
        feats = self.psi.m.get_intermediate_layers(x, n=list(self.layers))
        if self.emb is None or y is None:
            return torch.cat([h(f.transpose(1, 2)).flatten(1) for h, f in zip(self.heads, feats)], 1)
        outs = []
        for h, e, f in zip(self.heads, self.emb, feats):
            hid = h[:4](f.transpose(1, 2))                       # [N,256,T]
            o = h[4](hid)[:, 0] + (e(y)[:, :, None] * hid).sum(1) / 16.0
            outs.append(o)
        return torch.cat(outs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "enc", "none", "noise", "paired", "recon"], required=True,
                    help="noise: decoder noise-robustness baseline, L1(D(E(x)+sigma*eps), D_0(E(x)))")
    ap.add_argument("--paired_data", default="", help="paired mode (RobusTok-style SDEdit baseline): shard prefix, e.g. data/sdedit/t100")
    ap.add_argument("--lam_pair", type=float, default=1.0, help="weight of paired L1(D(z_sd), D_0(z0))")
    ap.add_argument("--noise", type=float, default=0.0, help="relative latent noise (enc / noise modes)")
    ap.add_argument("--ref_width", type=int, default=64, help="refiner base width (64 -> 6.8M, 128 -> 27M params)")
    ap.add_argument("--arch", choices=["decoder", "refiner"], default="decoder",
                    help="refiner: freeze the decoder, train a pixel-space CNN on its output")
    ap.add_argument("--loss", choices=["fd", "mmd", "gan"], default="fd")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam_anchor", type=float, default=1.0, help="weight of the anchor loss (0 = anchor-off ablation)")
    ap.add_argument("--target", choices=["adm_a"], default="adm_a")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16, help="batch for the distribution term")
    ap.add_argument("--bs_anchor", type=int, default=8)
    ap.add_argument("--queue", type=int, default=1024, help="detached past features for FD statistics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_real", type=int, default=0, help="use only the first n real target images of split A (0 = all 5k)")
    ap.add_argument("--n_gen", type=int, default=0, help="use only the first n cached generated latents (0 = all)")
    ap.add_argument("--save_every", type=int, default=0, help="also save decoder_step{k}.pt every k steps")
    ap.add_argument("--target_npz", default="", help="npz with key 'images' (uint8 NHWC): real target for L_dist (default: ADM split A)")
    ap.add_argument("--anchor_npz", default="", help="npz with key 'images': real images for the anchor loss (default: ImageNet-256 stream)")
    ap.add_argument("--cond", action="store_true", help="GAN only: class-conditional projection discriminator (real labels: ConvNeXt pseudo-labels)")
    ap.add_argument("--crop", type=int, default=0, help="GAN only: also show the discriminator random crops of this size (zoomed)")
    ap.add_argument("--crop_p", type=float, default=0.5, help="probability of the zoomed-crop view per step")
    ap.add_argument("--gen_pool", default="cfg1.5", help="generated-latent pool(s) in data/gen_train, comma-separated")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    vae = load_vae()
    vae.requires_grad_(False)
    dec0 = copy.deepcopy(vae)  # frozen original decoder
    if args.arch == "decoder":
        params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
        for p in params:
            p.requires_grad_(True)
        vae.enable_gradient_checkpointing()
        vae.decoder.train()

        def dec_train(z):
            return vae.decode(z).sample.float()
    else:
        w = args.ref_width
        refiner = Refiner((w, 2 * w, 4 * w)).cuda().train()
        params = list(refiner.parameters())
        print("refiner params (M):", sum(p.numel() for p in params) / 1e6, flush=True)

        def dec_train(z):
            with torch.no_grad():
                x0 = dec0.decode(z).sample.float()
            return refiner(x0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))

    psi = FeatNet(DINO)
    if args.anchor_npz:
        _anc = np.load(args.anchor_npz)["images"]
        _rng = np.random.default_rng(args.seed)

        def _anchor_iter():
            while True:
                yield torch.from_numpy(_anc[_rng.integers(0, len(_anc), max(args.bs, args.bs_anchor))])
        loader = _anchor_iter()
    else:
        loader = real_loader(max(args.bs, args.bs_anchor), seed=args.seed)

    # clean real target: ADM reference split A
    if args.target_npz:
        real_imgs = np.load(args.target_npz)["images"]
    else:
        split = np.load("data/adm_split.npz")["A"]
        if args.n_real:
            split = np.random.default_rng(0).permutation(split)[: args.n_real]
        real_imgs = np.load("data/VIRTUAL_imagenet256_labeled.npz")["arr_0"][split]
    if args.target_npz and args.n_real:
        real_imgs = real_imgs[: args.n_real]
    with torch.no_grad():
        bank = torch.cat([psi(from_uint8(real_imgs[i:i + 250])) for i in range(0, len(real_imgs), 250)])
    mu_r, c_r = bank.double().mean(0).float(), torch.cov(bank.double().T).float()
    sqrt_c_r = sqrtm_psd(c_r.double()).float()
    scale = torch.trace(c_r).item()  # FD normalised by total real variance

    if args.loss == "gan":
        disc = TokenDisc(psi, n_classes=1000 if args.cond else 0).cuda()
        opt_d = torch.optim.Adam(list(disc.heads.parameters()) + (list(disc.emb.parameters()) if disc.emb is not None else []), lr=args.lr_d, betas=(0.0, 0.99))

    zgen = load_gen_train(args.gen_pool) if args.mode == "gen" else None
    ygen = load_gen_train_labels(args.gen_pool) if (args.mode == "gen" and args.cond) else None
    yreal = torch.from_numpy(np.load("data/proc/adm_A_pseudolabels.npy").astype(np.int64)) if args.cond else None
    cur_y = [None, None]  # labels of the current generated / real batch
    if zgen is not None and args.n_gen:
        zgen = zgen[torch.randperm(len(zgen), generator=torch.Generator().manual_seed(0))[: args.n_gen]]
    if args.mode == "paired":
        import glob as _glob
        pf = sorted(_glob.glob(args.paired_data + "_r*.npz"))
        assert pf, args.paired_data
        pd = [np.load(f) for f in pf]
        pz0 = torch.from_numpy(np.concatenate([d["z0"] for d in pd]))
        pzsd = torch.from_numpy(np.concatenate([d["z_sd"] for d in pd]))
        print("paired samples:", len(pz0), flush=True)

    def src_latents():
        nonlocal_pair[0] = None
        if args.mode == "gen":
            ii = torch.randint(len(zgen), (args.bs,))
            if ygen is not None:
                cur_y[0] = ygen[ii].cuda()
            return zgen[ii].cuda().float()
        if args.mode == "paired":
            idx = torch.randint(len(pzsd), (args.bs,))
            nonlocal_pair[0] = pz0[idx].cuda().float()
            return pzsd[idx].cuda().float()
        with torch.no_grad():
            z = vae.encode(from_uint8(next(loader).numpy())[: args.bs]).latent_dist.sample()
            return z + args.noise * z.std() * torch.randn_like(z)

    nonlocal_pair = [None]
    if args.mode == "recon":
        import lpips
        lp = lpips.LPIPS(net="vgg").cuda().eval().requires_grad_(False)
    queue = torch.zeros(0, mu_r.numel(), device="cuda")
    if args.mode in ("gen", "enc") and args.loss == "fd":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            while len(queue) < args.queue:
                queue = torch.cat([queue, psi(dec_train(src_latents()).clamp(-1, 1))])
    log = open(f"{args.out}/log.jsonl", "w")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x = from_uint8(next(loader).numpy())
        with torch.no_grad():
            z_a = vae.encode(x[: args.bs_anchor]).latent_dist.sample()
            target = x[: args.bs_anchor] if args.mode == "recon" else dec0.decode(z_a).sample
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_a = dec_train(z_a)
            l_anchor = (y_a - target).abs().mean()
        if args.mode == "recon":  # standard decoder fine-tuning toward x (SD-VAE ft recipe): L1 + LPIPS
            l_anchor = l_anchor + lp(y_a.float().clamp(-1, 1), target.float()).mean()
        loss = args.lam_anchor * l_anchor
        l_dist, l_d = torch.zeros(()), torch.zeros(())
        if args.mode == "noise":
            with torch.no_grad():
                z_n = vae.encode(x[: args.bs]).latent_dist.sample()
                t_n = dec0.decode(z_n).sample
                z_n = z_n + args.noise * z_n.std() * torch.randn_like(z_n)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_dist = (dec_train(z_n) - t_n).abs().mean()
            loss = loss + args.lam * l_dist
        elif args.mode != "none":
            z_s = src_latents()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = dec_train(z_s).clamp(-1, 1)
            if args.mode == "paired":
                with torch.no_grad():
                    t_p = dec0.decode(nonlocal_pair[0]).sample.clamp(-1, 1)
                l_pair = (y - t_p).abs().mean()
                loss = loss + args.lam_pair * l_pair
            if args.loss == "gan":
                ri = np.random.randint(len(real_imgs), size=args.bs)
                xr = from_uint8(real_imgs[ri])
                if yreal is not None:
                    cur_y[1] = yreal[ri].cuda()
                if args.crop and np.random.rand() < args.crop_p:  # zoomed local view, identical for real and fake
                    yv = rand_crop_zoom(y, args.crop, gen)
                    xr = rand_crop_zoom(xr, args.crop, gen)
                else:
                    yv = y
                # discriminator step (hinge)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d_real = disc(diff_augment(xr, gen), cur_y[1]).float()
                    d_fake = disc(diff_augment(yv.detach(), gen), cur_y[0]).float()
                l_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                opt_d.zero_grad(set_to_none=True)
                l_d.backward()
                opt_d.step()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_dist = -disc(diff_augment(yv, gen), cur_y[0]).float().mean()
            else:
                f = psi(y)
                if args.loss == "fd":
                    l_dist = frechet_loss(torch.cat([queue, f]), mu_r, c_r, sqrt_c_r) / scale
                    queue = torch.cat([queue, f.detach()])[-args.queue:]
                else:
                    l_dist = mmd_loss(f, bank[torch.randint(len(bank), (256,), device="cuda")])
            loss = loss + args.lam * l_dist
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            rec = {"step": step, "anchor": l_anchor.item(), "dist": float(l_dist), "d": float(l_d),
                   "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if args.save_every and step % args.save_every == 0 and step != args.steps and args.arch == "decoder":
            torch.save({k: v.cpu() for k, v in vae.state_dict().items() if k.startswith(("decoder.", "post_quant_conv."))},
                       f"{args.out}/decoder_step{step}.pt")
    if args.arch == "decoder":
        sd = {k: v.cpu() for k, v in vae.state_dict().items() if k.startswith(("decoder.", "post_quant_conv."))}
    else:
        sd = {"refiner." + k: v.cpu() for k, v in refiner.state_dict().items()}
    torch.save(sd, f"{args.out}/decoder.pt")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
