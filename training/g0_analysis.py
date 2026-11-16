"""G0 analysis: is the generated-latent distribution off the encoder manifold, and does it matter?"""

import glob
import json

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sst

import fid


def roc_auc_score(y, s):
    """Mann-Whitney AUC (ties averaged)."""
    r = sst.rankdata(s)
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
from g0_process import load_set

SETS = ["real_sample", "real_mean", "m75_cfg1.0", "m75_cfg1.5", "all_cfg1.5"]


def load_proc(name):
    fs = sorted(glob.glob(f"data/proc/{name}_r*.npz"))
    ds = [np.load(f) for f in fs]
    order = np.argsort(np.concatenate([d["idx"] for d in ds]))
    return {k: np.concatenate([d[k] for d in ds])[order] for k in ["Pz", "f_D", "f_DP", "mse_img"]}


def cached_feats(path, images_fn):
    try:
        return np.load(path)
    except FileNotFoundError:
        f = fid.features(images_fn())
        np.save(path, f)
        return f


def radial_power(z, nb=16):
    """z: [N,C,H,W] -> [C, nb] mean radial power spectrum."""
    F = np.fft.fftshift(np.abs(np.fft.fft2(z - z.mean((2, 3), keepdims=True))) ** 2, axes=(2, 3)).mean(0)
    H = z.shape[-1]
    yy, xx = np.mgrid[:H, :H] - H // 2
    r = np.sqrt(yy ** 2 + xx ** 2)
    bins = np.minimum((r / (H // 2) * nb).astype(int), nb - 1)
    return np.stack([[F[c][bins == b].mean() for b in range(nb)] for c in range(z.shape[1])])


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def clf_auc(a, b, mu, sd, seed=0, epochs=30):
    """Train a small CNN to separate a (label 0) from b (label 1); return held-out AUC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    x = np.concatenate([a[:n], b[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    p = rng.permutation(2 * n)
    x, y = (x[p] - mu) / sd, y[p]
    nt = int(0.8 * len(x))
    xt, yt = torch.tensor(x[:nt]).float().cuda(), torch.tensor(y[:nt]).float().cuda()
    xv = torch.tensor(x[nt:]).float().cuda()
    m = Clf().cuda()
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (nt // 256))
    for _ in range(epochs):
        perm = torch.randperm(nt, device="cuda")
        for i in range(0, nt - 255, 256):
            j = perm[i:i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(m(xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    with torch.no_grad():
        s = torch.cat([m(xv[i:i + 1000]) for i in range(0, len(xv), 1000)]).cpu().numpy()
    return float(roc_auc_score(y[nt:], s))


def main():
    out = {}
    Z = {s: load_set(s).astype(np.float32) for s in SETS}
    P = {s: load_proc(s) for s in SETS}
    real = np.load("data/real75.npz")

    # 1. per-channel moments
    out["channel"] = {s: {"mean": Z[s].mean((0, 2, 3)).tolist(), "std": Z[s].std((0, 2, 3)).tolist(),
                          "skew": sst.skew(Z[s].transpose(1, 0, 2, 3).reshape(4, -1), axis=1).tolist(),
                          "kurt": sst.kurtosis(Z[s].transpose(1, 0, 2, 3).reshape(4, -1), axis=1).tolist()}
                      for s in SETS}

    # 2. radial spectrum, relative to real posterior samples
    ref = radial_power(Z["real_sample"])
    out["spectrum_ratio_vs_real_sample"] = {s: (radial_power(Z[s]) / ref).tolist() for s in SETS}

    # 3. cycle residual ||E(D(z)) - z||^2 / ||z||^2 and image change ||D(z) - D(E(D(z)))||^2
    res = {s: ((P[s]["Pz"] - Z[s]) ** 2).sum((1, 2, 3)) / (Z[s] ** 2).sum((1, 2, 3)) for s in SETS}
    q = lambda v: {k: float(f(v)) for k, f in [("mean", np.mean), ("p10", lambda a: np.percentile(a, 10)),
                                               ("median", np.median), ("p90", lambda a: np.percentile(a, 90))]}
    out["cycle_residual"] = {s: q(res[s]) for s in SETS}
    out["img_mse_D_vs_DP"] = {s: q(P[s]["mse_img"]) for s in SETS}
    # second-order: residual after projecting again (does P converge?)
    lab = lambda a, b: np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    out["residual_scalar_auc_vs_real_sample"] = {
        s: float(roc_auc_score(lab(res["real_sample"], res[s]), np.concatenate([res["real_sample"], res[s]])))
        for s in SETS if s != "real_sample"}

    # 4. CNN real-vs-gen AUC (class-matched sets only)
    mu = Z["real_sample"].mean((0, 2, 3), keepdims=True)[0]
    sd = Z["real_sample"].std((0, 2, 3), keepdims=True)[0]
    R = lambda s: Z[s] - P[s]["Pz"]
    pairs = {
        "control: real_mean vs real_sample": (Z["real_mean"], Z["real_sample"]),
        "raw: real_sample vs gen cfg1.0": (Z["real_sample"], Z["m75_cfg1.0"]),
        "raw: real_sample vs gen cfg1.5": (Z["real_sample"], Z["m75_cfg1.5"]),
        "projected: P(real_sample) vs P(gen cfg1.5)": (P["real_sample"]["Pz"], P["m75_cfg1.5"]["Pz"]),
        "residual: real_sample vs gen cfg1.5": (R("real_sample"), R("m75_cfg1.5")),
        "residual: real_sample vs gen cfg1.0": (R("real_sample"), R("m75_cfg1.0")),
    }
    out["cnn_auc"] = {}
    for k, (a, b) in pairs.items():
        sd_k = sd if not k.startswith("residual") else a.std((0, 2, 3), keepdims=True)[0]
        mu_k = mu if not k.startswith("residual") else a.mean((0, 2, 3), keepdims=True)[0]
        out["cnn_auc"][k] = clf_auc(a, b, mu_k, sd_k)
        print(k, out["cnn_auc"][k], flush=True)

    # 5. FD: D(z) vs D(E(D(z)))
    f_real = cached_feats("data/proc/feats_real75_orig.npy", lambda: real["images"])
    f_adm = cached_feats("data/proc/feats_adm_ref.npy",
                         lambda: np.load("data/VIRTUAL_imagenet256_labeled.npz")["arr_0"])
    s_real, s_adm = fid.stats(f_real), fid.stats(f_adm)
    out["fd"] = {}
    for s in SETS:
        r = s_adm if s.startswith("all") else s_real
        out["fd"][s] = {"ref": "adm_ref_10k" if s.startswith("all") else "real75_orig_10k",
                        "D(z)": fid.frechet(fid.stats(P[s]["f_D"]), r),
                        "D(E(D(z)))": fid.frechet(fid.stats(P[s]["f_DP"]), r)}
        print(s, out["fd"][s], flush=True)

    json.dump(out, open("results/g0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
