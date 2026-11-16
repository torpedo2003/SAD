"""Build polished LaTeX table bodies for the paper from results/*.json (missing entries -> '--').

Conventions: lower-is-better metrics get a colored delta vs. the original decoder D_0; best value per
column is bold; rows of our method get a light background (\\rowcolor{oursbg}); groups start with an
italic header row spanning the table.
"""

import json
import os

import numpy as np

OUT = "paper/tables"
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = f"results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def val(names, s, k):
    vs = []
    if s == "fid50kb":  # FD_CLIP and P/R against ADM split B, disjoint from the training target
        p = [f"results/fid50k_b/{'orig' if n == 'v3/orig' else n.replace('/', '_')}.json" for n in names]
        vs = [json.load(open(q))["dit50k_cfg1.5"][k] for q in p if os.path.exists(q)]
        return (float(np.mean(vs)), None) if vs else (None, None)
    if s == "fid50k":  # FID-50K exists only for some seeds: average over the ones that were run
        ps = [f"results/fid50k/{'orig' if n == 'v3/orig' else n.replace('/', '_')}.json" for n in names]
        vs = [json.load(open(p))["dit50k_cfg1.5"][k or "fid50k"] for p in ps if os.path.exists(p)]
        if not vs:
            return None, None
        return float(np.mean(vs)), (float(np.std(vs, ddof=1)) if len(vs) > 1 else None)
    for n in names:
        if s == "fid50k":
            p = f"results/fid50k/{n.replace('/', '_')}.json"
            v = json.load(open(p)).get("dit50k_cfg1.5", {}).get("fid50k") if os.path.exists(p) else None
        else:
            d = load(n)
            v = d.get("recon", {}).get(k) if s == "recon" else d.get(s, {}).get(k)
        if v is None:
            return None, None
        vs.append(v)
    return float(np.mean(vs)), (float(np.std(vs, ddof=1)) if len(vs) > 1 else None)


def seeds(base):
    return [f"v3/{base}", f"v3/{base}_s1", f"v3/{base}_s2"]


ORIG = ["v3/orig"]


class Col:
    def __init__(self, s, k, fmt, lower=True, std=False, delta=False):
        self.s, self.k, self.fmt, self.lower, self.std, self.delta = s, k, fmt, lower, std, delta


def render(rows, cols, ncols_total):
    """rows: list of ('group', title) | ('row', label_cells(list), names, ours(bool))."""
    data = [(r, [val(r[2], c.s, c.k) for c in cols]) for r in rows if r[0] == "row"]
    best = []
    for j, c in enumerate(cols):
        vs = [v[j][0] for _, v in data if v[j][0] is not None]
        best.append((min(vs) if c.lower else max(vs)) if vs else None)
    base = [val(ORIG, c.s, c.k)[0] for c in cols]
    out = []
    for r in rows:
        if r[0] == "group":
            out.append(r"\midrule" + "\n" + r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncols_total, r[1]))
            continue
        _, labels, names, ours = r
        cells = list(labels)
        for j, c in enumerate(cols):
            m, sd = val(names, c.s, c.k)
            if m is None:
                cells.append("--")
                continue
            s = c.fmt.format(m)
            if best[j] is not None and abs(m - best[j]) < 1e-9:
                s = r"\textbf{" + s + "}"
            if c.std and sd is not None:
                s += r"{\tiny$\,\pm$" + c.fmt.format(sd) + "}"
            if c.delta and base[j] is not None and names != ORIG:
                d = m - base[j]
                good = (d < 0) == c.lower
                if abs(d) >= 0.5 * 10 ** -(len(c.fmt.split(".")[1].rstrip("f}")) if "." in c.fmt else 0):
                    s += (r"\,\gain{" if good else r"\,\loss{") + ("+" if d > 0 else "$-$") + c.fmt.format(abs(d)) + "}"
            cells.append(s)
        out.append((r"\rowcolor{oursbg} " if ours else "") + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


# ---------------- Table: main ----------------
main_cols = [Col("all_cfg1.5", "fid", "{:.2f}", std=True), Col("fid50k", "", "{:.2f}", delta=True),
             Col("all_cfg1.5", "fd_clip", "{:.1f}", delta=True), Col("all_cfg1.5", "fd_dino", "{:.1f}"),
             Col("all_cfg1.5", "precision", "{:.3f}", lower=False), Col("all_cfg1.5", "recall", "{:.3f}", lower=False),
             Col("recon", "rfid", "{:.2f}", delta=True)]
main_rows = [
    ("row", [r"Original decoder $D_0$", "--"], ORIG, False),
    ("row", [r"Anchor only ($\lambda{=}0$)", "--"], ["v3/none"], False),
    ("group", r"Distribution loss on encoder latents $E(x)$"),
    ("row", ["GAN", "$E(x)$"], seeds("enc_gan0.05"), False),
    ("row", ["MMD", "$E(x)$"], seeds("enc_mmd1"), False),
    ("row", ["FD", "$E(x)$"], ["v3/enc_fd10"], False),
    ("group", r"\method: distribution loss on generated latents $\zgen$"),
    ("row", ["GAN", r"$\zgen$"], seeds("gen_gan0.05"), True),
    ("row", ["MMD", r"$\zgen$"], seeds("gen_mmd1"), True),
    ("row", ["FD", r"$\zgen$"], ["v3/gen_fd10", "v3/gen_fd10_s1"], True),
]
open(f"{OUT}/main.tex", "w").write(render(main_rows, main_cols, 9))

# ---------------- Table: transfer ----------------
TR = [("all_cfg1.5", r"cfg 1.5, 50 steps (train)"), ("all_cfg1.5s20", "cfg 1.5, 20 steps"), ("all_cfg1.0", "cfg 1.0, 50 steps"),
      ("all_cfg4.0", "cfg 4.0, 50 steps")]
rows = []
for s, label in TR:
    cells = [label]
    for k, fmt in (("fid", "{:.2f}"), ("fd_clip", "{:.1f}")):
        b = val(ORIG, s, k)[0]
        cells.append("--" if b is None else fmt.format(b))
        for ns in (seeds("gen_gan0.05"),):
            m = val(ns, s, k)[0]
            if m is None or b is None:
                cells.append("--")
                continue
            rel = 100 * (m - b) / b
            if round(rel) == 0:
                cells.append(fmt.format(m) + r"\,{\color{gray}\scriptsize $\pm$0\%}")
            else:
                cells.append(fmt.format(m) + (r"\,\gain{" if rel < 0 else r"\,\loss{") + f"{rel:+.0f}\\%".replace("-", "$-$") + "}")
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/transfer.tex", "w").write("\n".join(rows) + "\n")

# ---------------- Table: baselines ----------------
base_cols = [Col("all_cfg1.5", "fid", "{:.2f}", delta=True), Col("all_cfg1.5", "fd_clip", "{:.1f}"),
             Col("all_cfg1.5s20", "fid", "{:.2f}"), Col("all_cfg1.0", "fid", "{:.2f}"), Col("recon", "rfid", "{:.2f}")]
base_rows = [
    ("row", ["Original decoder $D_0$"], ORIG, False),
    ("group", "Robustness to latent perturbations"),
    ("row", [r"Noise aug., $\sigma{=}0.1$"], ["base/noise0.1"], False),
    ("row", [r"Noise aug., $\sigma{=}0.2$"], ["base/noise0.2"], False),
    ("row", [r"Noise aug., $\sigma{=}0.4$"], ["base/noise0.4"], False),
    ("row", [r"Noisy $E(x)$ + GAN"], ["base/enc_noise0.2_gan0.05"], False),
    ("group", r"Paired post-training~\cite{qiu2025robustok} (SDEdit)"),
    ("row", [r"$t_0{=}100$, $\ell_1$"], ["base/robustok_t100_l1"], False),
    ("row", [r"$t_0{=}100$, $\ell_1$+GAN"], ["base/robustok_t100"], False),
    ("row", [r"$t_0{=}250$, $\ell_1$+GAN"], ["base/robustok_t250"], False),
    ("group", r"Pixel-space refiner after frozen $D_0$"),
    ("row", ["6.8M, GAN"], ["base/refiner_gan0.05"], False),
    ("row", ["6.8M, MMD"], ["base/refiner_mmd1"], False),
    ("row", ["27M, GAN"], ["base/refiner128_gan0.05"], False),
    ("group", r"Decoder adaptation on $\zgen$ (ours)"),
    ("row", [r"\method-GAN"], seeds("gen_gan0.05"), True),
    ("row", [r"\method-MMD"], seeds("gen_mmd1"), True),
]
open(f"{OUT}/baselines.tex", "w").write(render(base_rows, base_cols, 6))

# ---------------- Table: ablation ----------------
abl_cols = [Col("all_cfg1.5", "fid", "{:.2f}"), Col("all_cfg1.5", "fd_clip", "{:.1f}"),
            Col("all_cfg1.0", "fid", "{:.2f}"), Col("recon", "rfid", "{:.2f}")]
abl_rows = [
    ("row", [r"\method-GAN (default)"], ["v3/gen_gan0.05"], True),
    ("group", r"GAN weight $\lambda$"),
    ("row", [r"$\lambda{=}0.1$"], ["v3/gen_gan0.1"], False),
    ("row", [r"$\lambda{=}0.2$"], ["v3/gen_gan0.2"], False),
    ("group", "Optimization"),
    ("row", ["4k steps"], ["v3/gen_gan0.05_4k"], False),
    ("row", [r"lr $5{\times}10^{-6}$"], ["v3/gen_gan0.05_lr5e-6"], False),
    ("row", [r"lr $3{\times}10^{-5}$"], ["v3/gen_gan0.05_lr3e-5"], False),
    ("group", "Other distribution losses"),
    ("row", [r"MMD, $\lambda{=}1$"], ["v3/gen_mmd1"], False),
    ("row", [r"MMD, $\lambda{=}3$"], ["v3/gen_mmd3"], False),
    ("row", [r"MMD, $\lambda{=}10$"], ["v3/gen_mmd10"], False),
    ("row", [r"MMD, 4k steps"], ["v3/gen_mmd1_4k"], False),
    ("row", [r"FD, $\lambda{=}10$"], ["v3/gen_fd10"], False),
    ("row", [r"FD, $\lambda{=}30$"], ["v3/gen_fd30"], False),
    ("group", "Amount of training data"),
    ("row", ["500 real images"], ["data/nreal500"], False),
    ("row", ["1k real images"], ["data/nreal1000"], False),
    ("row", ["2k real images"], ["data/nreal2000"], False),
    ("row", ["1k generated latents"], ["data/ngen1000"], False),
    ("row", ["5k generated latents"], ["data/ngen5000"], False),
    ("row", ["10k generated latents"], ["data/ngen10000"], False),
]
open(f"{OUT}/ablation.tex", "w").write(render(abl_rows, abl_cols, 5))
for f in ("main", "transfer", "baselines", "ablation"):
    print(f"== {f}\n" + open(f"{OUT}/{f}.tex").read())

# ---------------- teaser (d): relative change of FID / FD_CLIP, SAD-GAN vs D_0 ----------------
TEASE = [("all_cfg1.5", "DiT, cfg 1.5 (train)"), ("all_cfg1.5s20", "DiT, 20 steps"), ("all_cfg4.0", "DiT, cfg 4.0"),
         ("all_cfg1.0", "DiT, cfg 1.0"), ("sit_cfg1.5", "SiT, cfg 1.5"), ("sit_cfg1.0", "SiT, cfg 1.0")]
lines = ["label fid clip"]
for s, label in TEASE:
    b1, b2 = val(ORIG, s, "fid")[0], val(ORIG, s, "fd_clip")[0]
    m1, m2 = val(seeds("gen_gan0.05"), s, "fid")[0], val(seeds("gen_gan0.05"), s, "fd_clip")[0]
    if None in (b1, b2, m1, m2):
        continue
    lines.append("{%s} %.1f %.1f" % (label, 100 * (b1 - m1) / b1, 100 * (b2 - m2) / b2))
open("paper/fig/teaser_bars.dat", "w").write("\n".join(lines) + "\n")
print(open("paper/fig/teaser_bars.dat").read())

# ---------------- Table: guidance stacking ----------------
def gi(name, s, k):
    if s == "all_cfg1.5":
        return val([f"v3/{name}"] if name == "orig" else seeds("gen_gan0.05"), s, k)[0]
    fs = ["orig"] if name == "orig" else ["sad", "sad_s1", "sad_s2"]
    vs = [json.load(open(f"results/gi_sweep_{f}.json")).get(s, {}).get(k) for f in fs if os.path.exists(f"results/gi_sweep_{f}.json")]
    return float(np.mean(vs)) if vs and None not in vs else None


GI = [("all_cfg1.5", "1.5", "--"), ("full_cfg1.75", "1.75", "--"), ("gi_cfg2.0", "2.0", r"\checkmark"),
      ("gi_cfg2.5", "2.5", r"\checkmark"), ("gi_cfg3.0", "3.0", r"\checkmark")]
rows = []
best = min(gi(n, s, "fid") or 1e9 for s, _, _ in GI for n in ("orig", "sad"))
for s, w, iv in GI:
    cells = [w, iv]
    for k, fmt in (("fid", "{:.2f}"), ("fd_clip", "{:.1f}")):
        b, m = gi("orig", s, k), gi("sad", s, k)
        cells.append("--" if b is None else fmt.format(b))
        if m is None or b is None:
            cells.append("--"); continue
        t = fmt.format(m)
        if k == "fid" and abs(m - best) < 1e-9:
            t = r"\textbf{" + t + "}"
        rel = 100 * (m - b) / b
        cells.append(t + (r"\,\gain{" if rel < 0 else r"\,\loss{") + f"{rel:+.0f}\\%".replace("-", "$-$") + "}")
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/guidance.tex", "w").write("\n".join(rows) + "\n")
print("== guidance\n" + open(f"{OUT}/guidance.tex").read())

# ---------------- Table: generators ----------------
def mean_or_none(names, st, k):
    names = [n for n in names if os.path.exists(f"results/{n}.json")]
    return val(names, st, k)[0] if names else None


GENS = [(r"$D_0$", ORIG, False), ("DiT", seeds("gen_gan0.05"), True),
        ("SiT", ["sit/gen_gan0.05", "sit/gen_gan0.05_s1", "sit/gen_gan0.05_s2"], True),
        ("DiT + SiT", ["mix/gen_gan0.05", "mix/gen_gan0.05_s1"], True)]
GCOLS = [("all_cfg1.5", "fid", "{:.2f}"), ("all_cfg1.5", "fd_clip", "{:.1f}"), ("sit_cfg1.5", "fid", "{:.2f}"),
         ("sit_cfg1.5", "fd_clip", "{:.1f}"), ("sit_cfg1.0", "fid", "{:.2f}"), ("sit_cfg1.0", "fd_clip", "{:.1f}"), ("recon", "rfid", "{:.2f}")]
rows = []
for label, names, ours in GENS:
    cells = [label]
    for st, k, fmt in GCOLS:
        m, b = mean_or_none(names, st, k), val(ORIG, st, k)[0]
        if m is None:
            cells.append("--"); continue
        t = fmt.format(m)
        if names != ORIG and b is not None:
            rel = 100 * (m - b) / b
            if round(rel) == 0:
                t += r"\,{\color{gray}\scriptsize $\pm$0\%}"
            else:
                t += (r"\,\gain{" if rel < 0 else r"\,\loss{") + f"{rel:+.0f}\\%".replace("-", "$-$") + "}"
        cells.append(t)
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/generators.tex", "w").write("\n".join(rows) + "\n")
# compact single-column version: DiT FID | SiT 1.5 FID, FD_CLIP | SiT 1.0 FID | rFID
rows = []
GENS0 = [(r"$D_0$", ORIG, False), ("DiT", ["v3/gen_gan0.05"], True), ("SiT", ["sit/gen_gan0.05"], True),
         ("DiT + SiT", ["mix/gen_gan0.05"], True)]
for label, names, ours in GENS0:
    cells = [label]
    for st, k, fmt in [("all_cfg1.5", "fid", "{:.2f}"), ("sit_cfg1.5", "fid", "{:.2f}"), ("sit_cfg1.5", "fd_clip", "{:.1f}"),
                       ("sit_cfg1.0", "fid", "{:.2f}"), ("recon", "rfid", "{:.2f}")]:
        m, b0 = mean_or_none(names, st, k), val(ORIG, st, k)[0]
        t = "--" if m is None else fmt.format(m)
        if False:
            rel = 100 * (m - b0) / b0
            t += r"\,{\color{gray}\scriptsize $\pm$0\%}" if round(rel) == 0 else (r"\,\gain{" if rel < 0 else r"\,\loss{") + f"{rel:+.0f}\\%".replace("-", "$-$") + "}"
        cells.append(t)
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/generators_small.tex", "w").write("\n".join(rows) + "\n")
print("== generators\n" + open(f"{OUT}/generators.tex").read())

# ---------------- Supplementary: full transfer for all losses; SiT variants ----------------
SUPS = [("all_cfg1.5", "cfg 1.5 (train)"), ("all_cfg1.5s20", "20 steps"), ("all_cfg1.0", "cfg 1.0"), ("all_cfg4.0", "cfg 4.0")]
rows = []
for label, names in [(r"$D_0$", ORIG), ("GAN", seeds("gen_gan0.05")), ("MMD", seeds("gen_mmd1")), ("FD", ["v3/gen_fd10", "v3/gen_fd10_s1"])]:
    cells = [label]
    for st, _ in SUPS:
        for k, fmt in (("fid", "{:.2f}"), ("fd_clip", "{:.1f}")):
            m = val(names, st, k)[0]
            cells.append("--" if m is None else fmt.format(m))
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/supp_transfer.tex", "w").write("\n".join(rows) + "\n")

rows = []
for label, names in [(r"$D_0$", ORIG), (r"GAN, $\lambda{=}0.05$ (3 seeds)", ["sit/gen_gan0.05", "sit/gen_gan0.05_s1", "sit/gen_gan0.05_s2"]),
                     (r"GAN, $\lambda{=}0.2$", ["sit/gen_gan0.2"]), ("GAN, 4k steps", ["sit/gen_gan0.05_4k"]),
                     (r"MMD, $\lambda{=}1$ (2 seeds)", ["sit/gen_mmd1", "sit/gen_mmd1_s1"]), (r"FD, $\lambda{=}10$", ["sit/gen_fd10"])]:
    cells = [label]
    for st, k, fmt in (("sit_cfg1.5", "fid", "{:.2f}"), ("sit_cfg1.5", "fd_clip", "{:.1f}"), ("sit_cfg1.0", "fid", "{:.2f}"),
                       ("sit_cfg1.0", "fd_clip", "{:.1f}"), ("recon", "rfid", "{:.2f}")):
        m = mean_or_none(names, st, k)
        cells.append("--" if m is None else fmt.format(m))
    rows.append(" & ".join(cells) + r" \\")
open(f"{OUT}/supp_sit.tex", "w").write("\n".join(rows) + "\n")
print("== supp\n" + open(f"{OUT}/supp_transfer.tex").read() + open(f"{OUT}/supp_sit.tex").read())

# ---------------- Supplementary: FID-50K full ----------------
def f50(names, k):
    vs = []
    for n in names:
        p = f"results/fid50k/{'orig' if n == 'v3/orig' else n.replace('/', '_')}.json"
        if os.path.exists(p):
            vs.append(json.load(open(p))["dit50k_cfg1.5"][k])
    if not vs:
        return "--"
    m = np.mean(vs)
    fmt = "{:.2f}" if k == "fid50k" else "{:.1f}" if k == "fd_clip" else "{:.3f}"
    s = fmt.format(m)
    if len(vs) > 1:
        s += r"{\tiny$\,\pm$" + fmt.format(np.std(vs, ddof=1)) + "}"
    return s


rows = []
for label, names, grp in [(r"$D_0$", ["v3/orig"], None), (r"Anchor only", ["v3/none"], None),
                          ("GAN on $E(x)$", ["v3/enc_gan0.05"], None), ("MMD on $E(x)$", ["v3/enc_mmd1"], None),
                          ("FD on $E(x)$", ["v3/enc_fd10"], None),
                          (r"\method-GAN (3 seeds)", seeds("gen_gan0.05"), 1), (r"\method-MMD", ["v3/gen_mmd1"], 1),
                          (r"\method-FD", ["v3/gen_fd10"], 1)]:
    cells = [label] + [f50(names, k) for k in ("fid50k", "fd_clip", "precision", "recall")]
    rows.append((r"\rowcolor{oursbg} " if grp else "") + " & ".join(cells) + r" \\")
open(f"{OUT}/supp_fid50k.tex", "w").write("\n".join(rows) + "\n")

# ---------------- Supplementary: separability AUC ----------------
A = json.load(open("results/viz_analysis.json"))["probe_auc"]
SETN = {"all_cfg1.5": "DiT, cfg 1.5 (train)", "all_cfg1.5s20": "DiT, 20 steps", "all_cfg1.0": "DiT, cfg 1.0",
        "all_cfg4.0": "DiT, cfg 4.0", "sit_cfg1.5": "SiT, cfg 1.5"}
DECN = {"orig": "$D_0$", "sad": r"\method (DiT)", "enc": "$\\Ldist$ on $E(x)$", "mmd": r"\method-MMD (DiT)", "sadsit": r"\method (SiT)"}
rows, last = [], None
for key, v in A.items():
    s, d = key.split("/")
    if last is not None and s != last:
        rows.append(r"\midrule")
    lab = SETN[s] if s != last else ""
    last = s
    rows.append(f"{lab} & {DECN[d]} & " + " & ".join(f"{v[sp]:.3f}" for sp in ("inception", "clip", "dino")) + r" \\")
open(f"{OUT}/supp_auc.tex", "w").write("\n".join(rows) + "\n")
print("== fid50k\n" + open(f"{OUT}/supp_fid50k.tex").read() + "== auc\n" + open(f"{OUT}/supp_auc.tex").read())

# ---------------- Main: compact baselines (best config per family) + inference cost ----------------
cb_cols = [Col("all_cfg1.5", "fid", "{:.2f}", delta=True), Col("all_cfg1.5", "fd_clip", "{:.1f}"),
           Col("all_cfg1.5s20", "fid", "{:.2f}"), Col("all_cfg1.0", "fid", "{:.2f}"), Col("recon", "rfid", "{:.2f}")]
cb_rows = [
    ("row", ["Original decoder $D_0$", "--"], ORIG, False),
    ("row", [r"Noise-robust~\cite{zheng2025rae}", "0"], ["base/noise0.1"], False),
    ("row", [r"Paired SDEdit~\cite{qiu2025robustok}", "0"], ["base/robustok_t250"], False),
    ("row", ["Pixel refiner", "+15\\%"], ["base/refiner_gan0.05"], False),
    ("row", [r"\method-GAN", "0"], seeds("gen_gan0.05"), True),
    ("row", [r"\method-MMD", "0"], seeds("gen_mmd1"), True),
]
open(f"{OUT}/baselines_compact.tex", "w").write(render(cb_rows, cb_cols, 7))
print("== compact\n" + open(f"{OUT}/baselines_compact.tex").read())


# ---------------- Main table v2: richer (10k + 50k + transfer + recon), baselines merged ----------------
mv_cols = [Col("all_cfg1.5", "fid", "{:.2f}"), Col("all_cfg1.5", "fd_clip", "{:.1f}"),
           Col("all_cfg1.5", "fd_dino", "{:.1f}"), Col("all_cfg1.5", "precision", "{:.3f}", lower=False),
           Col("all_cfg1.5", "recall", "{:.3f}", lower=False),
           Col("fid50k", "fid50k", "{:.2f}"), Col("fid50kb", "fd_clip", "{:.1f}"),
           Col("fid50kb", "precision", "{:.3f}", lower=False), Col("fid50kb", "recall", "{:.3f}", lower=False),
           Col("all_cfg1.5s20", "fid", "{:.2f}"), Col("all_cfg1.0", "fid", "{:.2f}"),
           Col("recon", "rfid", "{:.2f}"), Col("recon", "psnr", "{:.2f}", lower=False)]
mv_rows = [
    ("row", [r"Original decoder $D_0$", "--"], ORIG, False),
    ("row", [r"Anchor only ($\lambda{=}0$)", "--"], ["v3/none"], False),
    ("group", r"Distribution loss on encoder latents $E(x)$"),
    ("row", ["GAN", "$E(x)$"], ["v3/enc_gan0.05"], False),
    ("row", ["MMD", "$E(x)$"], ["v3/enc_mmd1"], False),
    ("row", ["FD", "$E(x)$"], ["v3/enc_fd10"], False),
    ("group", r"Other decoder adaptations"),
    ("row", [r"Noise-robust~\cite{zheng2025rae}", r"$E(x){+}\epsilon$"], ["base/noise0.1"], False),
    ("row", [r"Paired SDEdit~\cite{qiu2025robustok}", "SDEdit"], ["base/robustok_t250"], False),
    ("row", [r"Pixel refiner", r"$\zgen$"], ["base/refiner_gan0.05"], False),
    ("group", r"\method: distribution loss on generated latents $\zgen$"),
    ("row", ["GAN", r"$\zgen$"], ["v3/gen_gan0.05"], True),
    ("row", ["MMD", r"$\zgen$"], ["v3/gen_mmd1"], True),
    ("row", ["FD", r"$\zgen$"], ["v3/gen_fd10"], True),
]
open(f"{OUT}/main_v2.tex", "w").write(render(mv_rows, mv_cols, 15))
print("== main_v2\n" + open(f"{OUT}/main_v2.tex").read())

# ---------------- Main: "is it just sharpening?" — post-processing, decoder swaps, standard decoder FT ----------------
def e3(key, m):
    p = "results/e3_baselines.json"
    d = json.load(open(p)) if os.path.exists(p) else {}
    return d.get(key, {}).get(m)


def from_run(name, m):
    d = load(name)
    if not d:
        return None
    return d.get("recon", {}).get(m) if m in ("rfid", "psnr") else d.get("all_cfg1.5", {}).get(m)


SH_ROWS = [("Original decoder $D_0$", "--", lambda m: from_run("v3/orig", m), False),
           (r"Unsharp mask (tuned)", "post-proc.", lambda m: e3("unsharp", m), False),
           (r"Additive noise (tuned)", "post-proc.", lambda m: e3("noise", m), False),
           (r"sd-vae-ft-mse~\cite{rombach2022ldm}", "decoder", lambda m: e3("ft_mse", m), False),
           (r"Consistency Decoder~\cite{openai2023consistencydecoder}", "decoder", lambda m: e3("consistency", m), False),
           (r"Fine-tune: $\ell_1$+LPIPS", r"$E(x)\!\to\!x$", lambda m: from_run("base/recon_lpips_lr1e-6", m), False),
           (r"Fine-tune: $\ell_1$+LPIPS+GAN", r"$E(x)\!\to\!x$", lambda m: from_run("base/recon_lpips_gan0.05_lr3e-6", m), False),
           (r"\method-GAN", r"$\zgen$", lambda m: from_run("v3/gen_gan0.05", m), True)]
SH_COLS = [("fid", "{:.2f}", True), ("fd_clip", "{:.1f}", True), ("precision", "{:.3f}", False),
           ("recall", "{:.3f}", False), ("rfid", "{:.2f}", True)]
vals = [[f(m) for m, _, _ in SH_COLS] for _, _, f, _ in SH_ROWS]
best = []
for j, (m, _, low) in enumerate(SH_COLS):
    vs = [v[j] for v in vals if v[j] is not None]
    best.append((min(vs) if low else max(vs)) if vs else None)
rows = []
for (lab, kind, _, ours), v in zip(SH_ROWS, vals):
    cells = [lab]
    for j, (m, fmt, _) in enumerate(SH_COLS):
        if v[j] is None:
            cells.append("--"); continue
        t = fmt.format(v[j])
        cells.append(r"\textbf{" + t + "}" if best[j] is not None and abs(v[j] - best[j]) < 1e-9 else t)
    rows.append((r"\rowcolor{oursbg} " if ours else "") + " & ".join(cells) + r" \\")
open(f"{OUT}/sharpen.tex", "w").write("\n".join(rows) + "\n")
print("== sharpen\n" + open(f"{OUT}/sharpen.tex").read())

# Module ablation: the distribution loss, its placement on z_gen, and the anchor, each removed in turn.
mod_cols = [Col("all_cfg1.5", "fid", "{:.2f}"), Col("all_cfg1.5", "fd_clip", "{:.1f}"),
            Col("recon", "rfid", "{:.2f}"), Col("recon", "psnr", "{:.2f}", lower=False)]
mod_rows = [("row", [r"$D_0$", "--", "--"], ["v3/orig"], False),
            ("row", [r"Anchor only", r"\ding{51}", "--"], ["v3/none"], False)]
for title, run in ((r"GAN", "gan0.05"), (r"MMD", "mmd1"), (r"FD", "fd10")):
    mod_rows += [("group", title),
                 ("row", [r"No anchor", "--", r"$\zgen$"], [f"v3/gen_{run}_noanc"], False),
                 ("row", [r"Loss on $E(x)$", r"\ding{51}", r"$E(x)$"], [f"v3/enc_{run}"], False),
                 ("row", [r"\method", r"\ding{51}", r"$\zgen$"], [f"v3/gen_{run}"], True)]
open(f"{OUT}/modules.tex", "w").write(render(mod_rows, mod_cols, 7))
print("== modules\n" + open(f"{OUT}/modules.tex").read())

# FDDA-style baseline (our re-implementation; Inception is part of its objective, so FID is not held out for it).
def _f50(n):
    return json.load(open(f"results/fid50k/{n}.json"))["dit50k_cfg1.5"]
FD_ROWS = [(r"$D_0$", "orig", "v3/orig", False), (r"\method-GAN", "v3_gen_gan0.05", "v3/gen_gan0.05", True),
           (r"FDDA-style, 2k steps", "fdda_2k_noanc", "fdda/2k_noanc", False),
           (r"FDDA-style, 8k steps", "fdda_8k_noanc", "fdda/8k_noanc", False),
           (r"FDDA-style + anchor, 8k steps", "fdda_8k_anc", "fdda/8k_anc", False)]
rows = []
for lab, f, r, ours in FD_ROWS:
    a, d = {**_f50(f), **{k: v for k, v in json.load(open(f'results/fid50k_b/{f}.json'))['dit50k_cfg1.5'].items() if k == 'fd_clip'}}, load(r)
    cells = [lab, f"{a['fid50k']:.2f}", f"{a['fd_clip']:.1f}", f"{d['all_cfg1.5']['fd_dino']:.1f}", f"{d['recon']['rfid']:.2f}", f"{d['recon']['psnr']:.2f}"]
    rows.append((r"\rowcolor{oursbg} " if ours else "") + " & ".join(cells) + r" \\")
open(f"{OUT}/fdda.tex", "w").write("\n".join(rows) + "\n")
print("== fdda\n" + open(f"{OUT}/fdda.tex").read())

# Generality across generators and domains: D_0 -> SAD for each setting.
import re as _re
def _tf(name, key):
    t = open(f"results/e1/{name}.txt").read()
    return float(_re.search(rf"^{key}: ([0-9.]+)", t, _re.M).group(1))
def _ar(a, b, fmt):
    return "--" if a is None or b is None else (fmt.format(a) + r"$\to$" + fmt.format(b))
def _p(n, s):
    return json.load(open(f"results/fid50k/{n}.json"))[s]
def _ext(n):
    return json.load(open(f"results/ext/coco_{n}.json"))
_pk = json.load(open("results/pickscore_coco.json"))
def _jb(n):
    return json.load(open(f"results/fid50k_b/{n}.json"))["ditoff_cfg1.5"]
def _pb(n, s):
    a = _p(n, s); a["fd_clip"] = json.load(open(f"results/fid50k_b/{n}.json"))[s]["fd_clip"]; return a
_o, _sg, _rg, _rm, _ro = _pb("orig", "dit50k_cfg1.5"), _pb("v3_gen_gan0.05", "dit50k_cfg1.5"), _pb("repa_gen_gan0.05", "repa_eval"), _pb("repa_gen_mmd1", "repa_eval"), _pb("repa_orig", "repa_eval")
_co, _v, _sit = _ext("orig"), load("v3/orig"), load("sit/gen_gan0.05")
def _c(k, n):
    g = _ext(n)
    return [_ar(_co["gen"]["fid"], g["gen"]["fid"], "{:.2f}"), _ar(_co["gen"]["fd_clip"], g["gen"]["fd_clip"], "{:.0f}"), _ar(_co["recon"]["rfid"], g["recon"]["rfid"], "{:.2f}"), f"{100 * _pk[k]['win_rate_vs_D0']:.0f}\\%"]
GEN = [
 ("group", "DiT-XL/2 on ImageNet (50k samples)"),
 (r"DPM-50, GAN", [_ar(_o["fid50k"], _sg["fid50k"], "{:.2f}"), _ar(_o["fd_clip"], _sg["fd_clip"], "{:.1f}"), _ar(2.63, load("v3/gen_gan0.05")["recon"]["rfid"], "{:.2f}"), "--"]),
 # Official DiT code and checkpoint (250-step DDPM, cfg 1.5 on three channels); SAD trained on a fresh official-code pool
 # (40k pool, seeds 2900-2912) disjoint from the 50k eval seeds; FID from the official ADM evaluator, FD_CLIP vs split B.
 (r"DDPM-250 (official), GAN", [_ar(_tf("ditoff_cfg1.5__orig", "FID"), _tf("ditoff_cfg1.5__d40_gen_gan0.05", "FID"), "{:.2f}"), _ar(_jb("ditoff_orig")["fd_clip"], _jb("ditoff_d40_gen_gan0.05")["fd_clip"], "{:.1f}"), _ar(2.63, json.load(open("results/ditoff40/gen_gan0.05.json"))["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"DDPM-250 (official), MMD", [_ar(_tf("ditoff_cfg1.5__orig", "FID"), _tf("ditoff_cfg1.5__d40_gen_mmd1", "FID"), "{:.2f}"), _ar(_jb("ditoff_orig")["fd_clip"], _jb("ditoff_d40_gen_mmd1")["fd_clip"], "{:.1f}"), _ar(2.63, json.load(open("results/ditoff40/gen_mmd1.json"))["recon"]["rfid"], "{:.2f}"), "--"]),
 ("group", "LightningDiT with VA-VAE on ImageNet (50k samples)"),
 (r"LightningDiT-XL, GAN", [_ar(_pb("ldit_orig", "ldit_eval")["fid50k"], _pb("ldit_gen_gan0.05", "ldit_eval")["fid50k"], "{:.2f}"), _ar(_pb("ldit_orig", "ldit_eval")["fd_clip"], _pb("ldit_gen_gan0.05", "ldit_eval")["fd_clip"], "{:.1f}"), _ar(load("ldit/orig")["recon"]["rfid"], load("ldit/gen_gan0.05")["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"LightningDiT-XL, MMD", [_ar(_pb("ldit_orig", "ldit_eval")["fid50k"], _pb("ldit_gen_mmd1", "ldit_eval")["fid50k"], "{:.2f}"), _ar(_pb("ldit_orig", "ldit_eval")["fd_clip"], _pb("ldit_gen_mmd1", "ldit_eval")["fd_clip"], "{:.1f}"), _ar(load("ldit/orig")["recon"]["rfid"], load("ldit/gen_mmd1")["recon"]["rfid"], "{:.2f}"), "--"]),
 ("group", "SiT on ImageNet"),
 (r"SiT-XL/2, GAN (10k)", [_ar(_v["sit_cfg1.5"]["fid"], _sit["sit_cfg1.5"]["fid"], "{:.2f}"), _ar(_v["sit_cfg1.5"]["fd_clip"], _sit["sit_cfg1.5"]["fd_clip"], "{:.1f}"), _ar(2.63, _sit["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"SiT+REPA, GAN (50k)", [_ar(_tf("repa_eval__orig", "FID"), _tf("repa_eval__gen_gan0.05", "FID"), "{:.2f}"), _ar(_ro["fd_clip"], _rg["fd_clip"], "{:.1f}"), _ar(2.63, load("repa/gen_gan0.05")["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"SiT+REPA, MMD (50k)", [_ar(_tf("repa_eval__orig", "FID"), _tf("repa_eval__gen_mmd1", "FID"), "{:.2f}"), _ar(_ro["fd_clip"], _rm["fd_clip"], "{:.1f}"), _ar(2.63, load("repa/gen_mmd1")["recon"]["rfid"], "{:.2f}"), "--"]),
 ("group", "REPA-E with E2E-VAE on ImageNet (50k samples)"),
 (r"SiT-XL/1, GAN", [_ar(_tf("repae_eval__orig", "FID"), _tf("repae_eval__sde_gen_gan0.05", "FID"), "{:.2f}"), _ar(json.load(open("results/fid50k_b/repae_orig.json"))["repae_eval"]["fd_clip"], json.load(open("results/fid50k_b/repae_sde_gen_gan0.05.json"))["repae_eval"]["fd_clip"], "{:.1f}"), _ar(load("repae/orig")["recon"]["rfid"], load("repae/sde_gen_gan0.05")["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"SiT-XL/1, GAN, ODE pool", [_ar(_tf("repae_eval__orig", "FID"), _tf("repae_eval__gen_gan0.05", "FID"), "{:.2f}"), _ar(json.load(open("results/fid50k_b/repae_orig.json"))["repae_eval"]["fd_clip"], json.load(open("results/fid50k_b/repae_gen_gan0.05.json"))["repae_eval"]["fd_clip"], "{:.1f}"), _ar(load("repae/orig")["recon"]["rfid"], load("repae/gen_gan0.05")["recon"]["rfid"], "{:.2f}"), "--"]),
 (r"SiT-XL/1, MMD, ODE pool", [_ar(_tf("repae_eval__orig", "FID"), _tf("repae_eval__gen_mmd1", "FID"), "{:.2f}"), _ar(json.load(open("results/fid50k_b/repae_orig.json"))["repae_eval"]["fd_clip"], json.load(open("results/fid50k_b/repae_gen_mmd1.json"))["repae_eval"]["fd_clip"], "{:.1f}"), _ar(load("repae/orig")["recon"]["rfid"], load("repae/gen_mmd1")["recon"]["rfid"], "{:.2f}"), "--"]),
 ("group", "PixArt-$\\alpha$ on COCO (10k samples)"),
 (r"GAN, $\lambda{=}0.01$", _c("g01", "gen_gan0.01_cc")), (r"GAN, $\lambda{=}0.05$", _c("gan_cc", "gen_gan0.05_cc")),
 (r"MMD, $\lambda{=}0.1$", _c("m01", "gen_mmd0.1_cc")), (r"MMD, $\lambda{=}0.3$", _c("m03", "gen_mmd0.3_cc")),
]
# Pick column for ImageNet rows: PickScore win rate vs D_0 with "a photo of a <class>" prompts (results/metrics_<domain>.json).
def _pw(dom, key):
    try:
        return f"{100 * json.load(open(f'results/metrics_{dom}.json'))[key]['pick_win']:.0f}\\%"
    except (FileNotFoundError, KeyError):
        return "--"
_PICK = {r"DPM-50, GAN": ("imagenet", "gen_gan0.05"), r"DDPM-250 (official), GAN": ("ditoff", "gan40"),
         r"DDPM-250 (official), MMD": ("ditoff", "mmd40"), r"LightningDiT-XL, GAN": ("ldit", "gan"), r"LightningDiT-XL, MMD": ("ldit", "mmd"),
         r"SiT-XL/2, GAN (10k)": ("sit", "gan"), r"SiT+REPA, GAN (50k)": ("repa", "gan"), r"SiT+REPA, MMD (50k)": ("repa", "mmd"),
         r"SiT-XL/1, GAN": ("repae", "sde"), r"SiT-XL/1, GAN, ODE pool": ("repae", "gan"), r"SiT-XL/1, MMD, ODE pool": ("repae", "mmd")}
GEN = [(r[0], r[1][:-1] + [_pw(*_PICK[r[0]])]) if r[0] in _PICK else r for r in GEN]
rows = []
for r in GEN:
    rows.append(r"\midrule" + "\n" + r"\multicolumn{5}{l}{\textit{%s}} \\" % r[1] if r[0] == "group" else r[0] + " & " + " & ".join(r[1]) + r" \\")
rows[0] = rows[0].replace(r"\midrule" + "\n", "")
open(f"{OUT}/generality.tex", "w").write("\n".join(rows) + "\n")
print("== generality\n" + open(f"{OUT}/generality.tex").read())

# Extra metrics (full list, including those that do not favour SAD) and the complete PixArt lambda curve.
_m = {d: json.load(open(f"results/metrics_{d}.json")) for d in ("imagenet", "coco", "ditoff", "ldit", "repae")}
def _fmt(r, k, f):
    return f.format(r[k]) if k in r else "--"
def _mrow(lab, r, ours=False):
    c = [lab, _fmt(r, "kid", "{:.2f}"), _fmt(r, "fd_siglip", "{:.1f}"), _fmt(r, "cmmd", "{:.3f}"), _fmt(r, "hps", "{:.3f}"), _fmt(r, "pick", "{:.2f}")]
    w = [(f"{100 * r[k]:.0f}\\%" if k in r else "--") for k in ("hps_win", "pick_win")]
    return (r"\rowcolor{oursbg} " if ours else "") + " & ".join(c + w) + r" \\"
rows = [r"\multicolumn{8}{l}{\textit{DiT-XL/2, DPM-Solver++ (50 steps)}} \\", _mrow(r"$D_0$", _m["imagenet"]["orig"]), _mrow(r"$\Ldist$ on $E(x)$, GAN", _m["imagenet"]["enc_gan0.05"]), _mrow(r"\method-GAN", _m["imagenet"]["gen_gan0.05"], True), _mrow(r"\method-MMD", _m["imagenet"]["gen_mmd1"], True),
        r"\midrule", r"\multicolumn{8}{l}{\textit{DiT-XL/2, official code (DDPM-250)}} \\", _mrow(r"$D_0$", _m["ditoff"]["orig"]), _mrow(r"$\Ldist$ on $E(x)$, GAN", _m["ditoff"]["enc_gan"]), _mrow(r"\method-GAN", _m["ditoff"]["gan40"], True), _mrow(r"\method-MMD", _m["ditoff"]["mmd40"], True),
        r"\midrule", r"\multicolumn{8}{l}{\textit{LightningDiT-XL with VA-VAE}} \\", _mrow(r"$D_0$", _m["ldit"]["orig"]), _mrow(r"$\Ldist$ on $E(x)$, GAN", _m["ldit"]["enc_gan"]), _mrow(r"\method-GAN", _m["ldit"]["gan"], True), _mrow(r"\method-MMD", _m["ldit"]["mmd"], True),
        r"\midrule", r"\multicolumn{8}{l}{\textit{REPA-E with E2E-VAE}} \\", _mrow(r"$D_0$", _m["repae"]["orig"]), _mrow(r"\method-GAN", _m["repae"]["sde"], True), _mrow(r"\method-GAN, ODE pool", _m["repae"]["gan"], True), _mrow(r"\method-MMD, ODE pool", _m["repae"]["mmd"], True),
        r"\midrule", r"\multicolumn{8}{l}{\textit{PixArt-$\alpha$ on COCO}} \\", _mrow(r"$D_0$", _m["coco"]["orig"]), _mrow(r"$\Ldist$ on $E(x)$, GAN", _m["coco"]["enc_gan0.05_cc"]), _mrow(r"\method-GAN, $\lambda{=}0.01$", _m["coco"]["gen_gan0.01_cc"], True), _mrow(r"\method-GAN, $\lambda{=}0.05$", _m["coco"]["gen_gan0.05_cc"], True), _mrow(r"\method-MMD, $\lambda{=}0.1$", _m["coco"]["gen_mmd0.1_cc"], True), _mrow(r"\method-MMD, $\lambda{=}1$", _m["coco"]["gen_mmd1_cc"], True), _mrow(r"DiT decoder, transferred", _m["coco"]["dit_transfer"]), _mrow(r"Real images (val2017)", _m["coco"]["real"])]
open(f"{OUT}/extra_metrics.tex", "w").write("\n".join(rows) + "\n")
rows = []
for lab, n, k in [(r"$D_0$", "orig", None), (r"GAN, $\lambda{=}0.005$", "gen_gan0.005_cc", "g005"), (r"GAN, $\lambda{=}0.01$", "gen_gan0.01_cc", "g01"), (r"GAN, $\lambda{=}0.02$", "gen_gan0.02_cc", "g02"), (r"GAN, $\lambda{=}0.05$", "gen_gan0.05_cc", "gan_cc"), (r"MMD, $\lambda{=}0.1$", "gen_mmd0.1_cc", "m01"), (r"MMD, $\lambda{=}0.3$", "gen_mmd0.3_cc", "m03"), (r"MMD, $\lambda{=}1$", "gen_mmd1_cc", "mmd_cc")]:
    g = _ext(n)
    w = f"{100 * _pk[k]['win_rate_vs_D0']:.0f}\\%" if k else "--"
    rows.append(" & ".join([lab, f"{g['gen']['fid']:.2f}", f"{g['gen']['fd_clip']:.1f}", f"{g['gen']['clip_score']:.2f}", f"{g['gen']['recall']:.3f}", f"{g['recon']['rfid']:.2f}", w]) + r" \\")
open(f"{OUT}/coco_lam.tex", "w").write("\n".join(rows) + "\n")

# Anchor-weight sweep, no-anchor tuning and small-lambda controls on encoder latents.
_sw = [("group", r"Anchor weight, \method-GAN"), (r"Weight 3", "v3/gen_gan0.05_anc3"), (r"Weight 1 (default)", "v3/gen_gan0.05"), (r"Weight 0.3", "v3/gen_gan0.05_anc0.3"), (r"Weight 0.1", "v3/gen_gan0.05_anc0.1"), (r"Weight 0", "v3/gen_gan0.05_noanc"),
       ("group", r"No anchor, \method-GAN"), (r"lr $3{\times}10^{-6}$", "v3/gen_gan0.05_noanc_lr3e-6"), (r"500 steps", "v3/gen_gan0.05_noanc_500"), (r"1k steps", "v3/gen_gan0.05_noanc_1k"),
       ("group", r"$\Ldist$ on $E(x)$"), (r"GAN, $\lambda{=}0.005$", "v3/enc_gan0.005"), (r"GAN, $\lambda{=}0.01$", "v3/enc_gan0.01"), (r"GAN, $\lambda{=}0.05$", "v3/enc_gan0.05"), (r"MMD, $\lambda{=}0.1$", "v3/enc_mmd0.1"), (r"MMD, $\lambda{=}0.3$", "v3/enc_mmd0.3"), (r"MMD, $\lambda{=}1$", "v3/enc_mmd1")]
rows = []
for r in _sw:
    if r[0] == "group":
        rows += ([] if not rows else [r"\midrule"]) + [r"\multicolumn{5}{l}{\textit{%s}} \\" % r[1]]
    else:
        d = load(r[1]); a = d["all_cfg1.5"]
        rows.append(f"{r[0]} & {a['fid']:.2f} & {a['fd_clip']:.1f} & {d['recon']['rfid']:.2f} & {d['recon']['psnr']:.2f}" + r" \\")
open(f"{OUT}/anchor_sweep.tex", "w").write("\n".join(rows) + "\n")

# Main-text tables: FID-50K over guidance scales; anchor-weight sweep.
rows = []
_oa, _sa = json.load(open("results/fid50k/orig.json")), json.load(open("results/fid50k/v3_gen_gan0.05.json"))
for c in ("1.25", "1.375", "1.5"):
    a, b = _oa[f"dit50k_cfg{c}"]["fid50k"], _sa[f"dit50k_cfg{c}"]["fid50k"]
    rows.append(f"{c} & {a:.2f} & {b:.2f} & {100 * (a - b) / a:.0f}\\% " + r"\\")
open(f"{OUT}/cfg_scan.tex", "w").write("\n".join(rows) + "\n")
rows = []
for lab, n, ours in ((r"3", "v3/gen_gan0.05_anc3", False), (r"1 (default)", "v3/gen_gan0.05", True), (r"0.3", "v3/gen_gan0.05_anc0.3", False), (r"0.1", "v3/gen_gan0.05_anc0.1", False), (r"0", "v3/gen_gan0.05_noanc", False)):
    d = load(n); a = d["all_cfg1.5"]
    rows.append((r"\rowcolor{oursbg} " if ours else "") + f"{lab} & {a['fid']:.2f} & {a['fd_clip']:.1f} & {d['recon']['rfid']:.2f} & {d['recon']['psnr']:.2f}" + r" \\")
open(f"{OUT}/anchor_main.tex", "w").write("\n".join(rows) + "\n")

# Main-text table: metrics that play no role in training, on both datasets.
rows = []
for ds, lab, k, ours in (("ditoff", r"DiT-XL/2, $D_0$", "orig", False), ("ditoff", r"DiT-XL/2, \method", "gan", True), ("ldit", r"LightningDiT, $D_0$", "orig", False), ("ldit", r"LightningDiT, \method", "gan", True), ("coco", r"PixArt-$\alpha$, $D_0$", "orig", False), ("coco", r"PixArt-$\alpha$, \method", "gen_gan0.01_cc", True)):
    r = _m[ds][k]
    rows.append((r"\rowcolor{oursbg} " if ours else "") + f"{lab} & {r['kid']:.2f} & {r['fd_siglip']:.1f} & {r['cmmd']:.3f}" + r" \\")
open(f"{OUT}/extra_main.tex", "w").write("\n".join(rows) + "\n")

# Second tokenizer: all LightningDiT decoders.
rows = []
for lab, n, ours in ((r"$D_0$", "orig", False), (r"\method-GAN", "gen_gan0.05", True), (r"\method-MMD", "gen_mmd1", True), (r"$\Ldist$ on $E(x)$, GAN", "enc_gan0.05", False), (r"\method-GAN, no anchor", "gen_gan0.05_noanc", False)):
    a = json.load(open(f"results/fid50k/ldit_{n}.json"))["ldit_eval"]; b = json.load(open(f"results/fid50k_b/ldit_{n}.json"))["ldit_eval"]; d = load(f"ldit/{n}")
    rows.append((r"\rowcolor{oursbg} " if ours else "") + f"{lab} & {a['fid50k']:.2f} & {b['fd_clip']:.1f} & {b['precision']:.3f} & {b['recall']:.3f} & {d['recon']['rfid']:.2f} & {d['recon']['psnr']:.2f}" + r" \\")
open(f"{OUT}/ldit.tex", "w").write("\n".join(rows) + "\n")

# System-level comparison on ImageNet 256x256 with guidance. Reported numbers are copied from the cited papers (FID-50K, IS;
# official ADM evaluator): ADM-G, LDM-4-G, VAR-d30, MAR-H, DiT-XL/2 from their papers; DG on DiT from Kim et al. 2023;
# SiT-XL/2 (IS 270.3) and REPA (1.42, 305.7) from their papers; LightningDiT 1.35/295.3 (VA-VAE paper); REPA-E 1.12/302.9
# (arXiv v3, class-balanced labels). Our rows: D_0 -> SAD on the same 50k latents, official ADM evaluator.
def _tfo(name, key):
    try:
        return _tf(name, key)
    except (FileNotFoundError, AttributeError):
        return None
REP = [
 (r"ADM-G~\cite{dhariwal2021adm}", 4.59, 186.7),
 (r"LDM-4-G~\cite{rombach2022ldm}", 3.60, 247.7),
 (r"VAR-d30~\cite{tian2024var}", 1.92, 323.1),
 (r"MAR-H~\cite{li2024mar}", 1.55, 303.7),
 (r"DiT-XL/2~\cite{peebles2023dit}", 2.27, 278.2),
 (r"\quad + DG~\cite{kim2023dg}", 1.83, 281.5),
 (r"SiT-XL/2~\cite{ma2024sit}", 2.06, 270.3),
 (r"\quad + REPA~\cite{yu2025repa}", 1.42, 305.7),
 (r"LightningDiT-XL~\cite{yao2025vavae}", 1.35, 295.3),
 (r"REPA-E~\cite{leng2025repae}$^\dagger$", 1.12, 302.9),
]
def _rf(path):
    try:
        return json.load(open(path))["recon"]["rfid"]
    except (FileNotFoundError, KeyError):
        return None
OURS = [  # name, D_0 sample batch, SAD sample batch, rFID of D_0, rFID of SAD
 (r"DiT-XL/2", "ditoff_cfg1.5__orig", "ditoff_cfg1.5__d40_gen_gan0.05", 2.63, _rf("results/ditoff40/gen_gan0.05.json")),
 (r"\quad + DG", "dgpaper__orig", "dgpaper__sad", 2.63, _rf("results/ditoff40/gen_gan0.05.json")),
 (r"SiT-XL/2 + REPA", "repa_eval__orig", "repa_eval__gen_mmd1", 2.63, _rf("results/repa/gen_mmd1.json")),
 (r"LightningDiT-XL", "ldit_eval__orig", "ldit_eval__gen_gan0.05", _rf("results/ldit/orig.json"), _rf("results/ldit/gen_gan0.05.json")),
 (r"REPA-E$^\dagger$", "repae_eval__orig", "repae_eval__sde_gen_gan0.05", _rf("results/repae/orig.json"), _rf("results/repae/sde_gen_gan0.05.json")),
]
rows = [rf"{n} & {f:.2f} & {i:.1f} & -- \\" for n, f, i in REP]
rows += [r"\midrule", r"\multicolumn{4}{l}{\textit{$D_0\!\to\!$\method on the same latents (ours)}} \\"]
for n, a, b, ra, rb in OURS:
    rows.append(r"\rowcolor{oursbg} " + n + " & " + _ar(_tfo(a, "FID"), _tfo(b, "FID"), "{:.2f}") + " & " + _ar(_tfo(a, "Inception Score"), _tfo(b, "Inception Score"), "{:.1f}") + " & " + _ar(ra, rb, "{:.2f}") + r" \\")
open(f"{OUT}/sota.tex", "w").write("\n".join(rows) + "\n")
print("== sota\n" + open(f"{OUT}/sota.tex").read())

# ======================= Restructured paper (11-11 evening): three main-text tables =======================
# Tab.1 (sota_full): system-level ImageNet 256x256. Reported rows copied from the cited papers' tables:
#   ADM-G, DiT-XL/2, DG on DiT (Kim et al. 2023, Tab. ImageNet 256); SiT-XL/2, MAR-H (945M), REPA, MDTv2, LightningDiT
#   (VA-VAE paper Tab. 3); VAR-d30 (VAR paper Tab. 1; sFID not reported); REPA-E (arXiv v3 Tab. 9, class-balanced labels).
# Our rows: released checkpoints sampled with their official recipes, decoded by D_0 and by SAD, ADM evaluator.
def _fmt(v, f):
    return "--" if v is None else f.format(v)
_KEYS = [("FID", "{:.2f}", True), ("sFID", "{:.2f}", True), ("Inception Score", "{:.1f}", False), ("Precision", "{:.2f}", False), ("Recall", "{:.2f}", False)]
OTHERS = [  # name, tokenizer, FID, sFID, IS, P, R
 (r"ADM-G~\cite{dhariwal2021adm}", "--", 4.59, 5.25, 186.7, 0.82, 0.52),
 (r"VAR-d30~\cite{tian2024var}", "VQVAE", 1.92, None, 323.1, 0.82, 0.59),
 (r"MAR-H~\cite{li2024mar}", "KL-16", 1.55, None, 303.7, 0.81, 0.62),
 (r"SiT-XL/2~\cite{ma2024sit}", "SD-VAE", 2.06, 4.50, 270.3, 0.82, 0.59),
 (r"MDTv2-XL/2~\cite{gao2023mdtv2}", "SD-VAE", 1.58, 4.52, 314.7, 0.79, 0.65),
]
_CLIP = {"ditoff_cfg1.5__orig": ("ditoff_orig", "ditoff_cfg1.5"), "ditoff_cfg1.5__d40_gen_gan0.05": ("ditoff_d40_gen_gan0.05", "ditoff_cfg1.5"),
         "dgpaper__orig": ("dgpaper_orig", "dgpaper"), "dgpaper__sad": ("dgpaper_sad", "dgpaper"),
         "repa_eval__orig": ("repa_orig", "repa_eval"), "repa_eval__gen_gan0.05": ("repa_gen_gan0.05", "repa_eval"),
         "ldit_eval__orig": ("ldit_orig", "ldit_eval"), "ldit_eval__gen_gan0.05": ("ldit_gen_gan0.05", "ldit_eval"),
         "repae_eval__orig": ("repae_orig", "repae_eval"), "repae_eval__gen_gan0.05": ("repae_gen_gan0.05", "repae_eval"),
         "repae_eval__sde_gen_gan0.05": ("repae_sde_gen_gan0.05", "repae_eval")}
def _clip(batch):
    try:
        f, k = _CLIP[batch]; return json.load(open(f"results/fid50k_b/{f}.json"))[k]["fd_clip"]
    except (KeyError, FileNotFoundError):
        return None
BASES = [  # name, tokenizer, reported (FID,sFID,IS,P,R), D_0 batch, SAD batch
 (r"DiT-XL/2~\cite{peebles2023dit}", "SD-VAE", (2.27, 4.60, 278.2, 0.83, 0.57), "ditoff_cfg1.5__orig", "ditoff_cfg1.5__d40_gen_gan0.05"),
 (r"DiT-XL/2 + DG~\cite{kim2023dg}", "SD-VAE", (1.83, 5.16, 281.5, 0.78, 0.64), "dgpaper__orig", "dgpaper__sad"),
 (r"SiT-XL/2 + REPA~\cite{yu2025repa}", "SD-VAE", (1.42, 4.70, 305.7, 0.80, 0.65), "repa_eval__orig", "repa_eval__gen_gan0.05"),
 (r"LightningDiT-XL~\cite{yao2025vavae}", "VA-VAE", (1.35, 4.15, 295.3, 0.79, 0.65), "ldit_eval__orig", "ldit_eval__gen_gan0.05"),
 (r"REPA-E~\cite{leng2025repae}$^\dagger$", "E2E-VAE", (1.12, 4.09, 302.9, 0.79, 0.66), "repae_eval__orig", "repae_eval__sde_gen_gan0.05"),
]
rows = [r"\multicolumn{8}{l}{\textit{Other generators (reported)}} \\"]
# FD_CLIP of other generators, computed by us vs split B: ADM-G from its released 50k samples, VAR-d30 and SiT-XL/2 from
# released weights with their official FID recipes (fdclip_images.py, gen_var_samples.py, gen_latents_sit_sde.py; our
# pytorch FID 4.58 / 1.95 / 2.07 vs reported 4.59 / 1.92 / 2.06). MAR-H, MDTv2: not computed.
_OCLIP = {"ADM-G": ("ext_fdclip", "adm_g"), "VAR-d30": ("ext_fdclip_var", "var_d30"), "SiT-XL/2": ("sitsde_orig", "sitsde")}
def _oclip(n):
    k = n.split("~")[0]
    if k not in _OCLIP:
        return None
    f, key = _OCLIP[k]; return json.load(open(f"results/fid50k_b/{f}.json"))[key]["fd_clip"]
for n, tok, *v in OTHERS:
    rows.append(f"{n} & {tok} & " + " & ".join(_fmt(x, f) for x, (_, f, _) in zip(v, _KEYS)) + f" & {_fmt(_oclip(n), '{:.1f}')}" + r" \\")
for n, tok, rep, a, b in BASES:
    ours_a = [_tfo(a, k) for k, _, _ in _KEYS]
    ours_b = [_tfo(b, k) for k, _, _ in _KEYS]
    rows += [r"\midrule", f"{n} & {tok} & " + " & ".join(r"\textcolor{black!55}{" + _fmt(x, f) + "}" for x, (_, f, _) in zip(rep, _KEYS)) + r" & \textcolor{black!55}{--} \\"]
    ca, cb = _clip(a), _clip(b)
    rows.append(r"\quad $D_0$, reproduced & & " + " & ".join(_fmt(x, f) for x, (_, f, _) in zip(ours_a, _KEYS)) + f" & {_fmt(ca, '{:.1f}')}" + r" \\")
    cells = []
    for x, y, (_, f, low) in zip(ours_a, ours_b, _KEYS):
        if y is None:
            cells.append("--"); continue
        half = 0.5 * 10 ** -int(f[-3])  # half a displayed unit, e.g. 0.005 for {:.2f}
        dx = (y - x) if x is not None and abs(y - x) >= half else 0.0  # raw difference; rounding-only changes count as none
        t = r"\textbf{" + f.format(y) + "}" if ((dx < 0) if low else (dx > 0)) else f.format(y)  # bold only when SAD is strictly better
        if abs(dx) > 1e-9:
            dx = float(f.format(y)) - float(f.format(x))
            better = (dx < 0) if low else (dx > 0)
            t += r"\,{\scriptsize\textcolor{" + ("gaincol" if better else "losscol") + "}{" + ("$-$" if dx < 0 else "+") + f.format(abs(dx)) + "}}"
        cells.append(t)
    if cb is None:
        cc = "--"
    else:
        cc = (r"\textbf{%.1f}" % cb) if ca is not None and round(cb, 1) < round(ca, 1) else "%.1f" % cb
        if ca is not None and abs(round(cb, 1) - round(ca, 1)) > 1e-9:
            d = round(cb, 1) - round(ca, 1)
            cc += r"\,{\scriptsize\textcolor{" + ("gaincol" if d < 0 else "losscol") + "}{" + ("$-$" if d < 0 else "+") + "%.1f" % abs(d) + "}}"
    rows.append(r"\rowcolor{oursbg} \quad + \method (ours) & & " + " & ".join(cells) + f" & {cc}" + r" \\")
open(f"{OUT}/sota_full.tex", "w").write("\n".join(rows) + "\n")
print("== sota_full\n" + open(f"{OUT}/sota_full.tex").read())

# Tab.2 (adapt): decoder-side alternatives on DiT-XL/2, 50-step DPM-Solver++, 10k samples vs split B.
def _r(name, m):
    return from_run(name, m)
ADAPT = [  # group, label, getter, ours
 (None, r"Original decoder $D_0$", lambda m: _r("v3/orig", m), False),
 ("Replacing or re-training the decoder", None, None, None),
 (None, r"sd-vae-ft-mse~\cite{rombach2022ldm}", lambda m: e3("ft_mse", m), False),
 (None, r"Consistency Decoder~\cite{openai2023consistencydecoder}", lambda m: e3("consistency", m), False),
 (None, r"Fine-tuning, $\ell_1$+LPIPS+GAN", lambda m: _r("base/recon_lpips_gan0.05_lr3e-6", m), False),
 ("Robustness to perturbed latents", None, None, None),
 (None, r"Noise-robust decoder~\cite{zheng2025rae}", lambda m: _r("base/noise0.1", m), False),
 (None, r"Paired post-training~\cite{qiu2025robustok,fang2026gar}", lambda m: _r("base/robustok_t250", m), False),
 ("Alignment to real images", None, None, None),
 (None, r"Pixel refiner after $D_0$", lambda m: _r("base/refiner_gan0.05", m), False),
 (None, r"FDDA~\cite{fang2026fdda}$^\ast$", lambda m: (json.load(open("results/fdda/2k_noanc.json")).get("recon", {}).get(m) if m in ("rfid", "psnr") else json.load(open("results/fdda/2k_noanc.json"))["all_cfg1.5"].get(m)), False),
 (None, r"\method-GAN (ours)", lambda m: _r("v3/gen_gan0.05", m), True),
 (None, r"\method-MMD (ours)", lambda m: _r("v3/gen_mmd1", m), True),
]
AC = [("fid", "{:.2f}", True), ("fd_clip", "{:.1f}", True), ("rfid", "{:.2f}", True), ("psnr", "{:.2f}", False)]
vals = {i: [g(m) for m, _, _ in AC] for i, (grp, lab, g, o) in enumerate(ADAPT) if g}
best = [(min if low else max)(v[j] for v in vals.values() if v[j] is not None) for j, (_, _, low) in enumerate(AC)]
second = [sorted({round(v[j], 4) for v in vals.values() if v[j] is not None}, reverse=not low)[1] for j, (_, _, low) in enumerate(AC)]
rows = []
for i, (grp, lab, g, o) in enumerate(ADAPT):
    if grp:
        rows += [r"\midrule", r"\multicolumn{5}{l}{\textit{" + grp + r"}} \\"]; continue
    cells = []
    for j, (m, f, _) in enumerate(AC):
        x = vals[i][j]; t = f.format(x)
        if abs(x - best[j]) < 1e-9: t = r"\textbf{" + t + "}"
        elif abs(round(x, 4) - second[j]) < 1e-9: t = r"\underline{" + t + "}"
        cells.append(t)
    if lab.startswith(r"\method-GAN"):
        rows.append(r"\midrule")
    rows.append((r"\rowcolor{oursbg} " if o else "") + lab + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/adapt.tex", "w").write("\n".join(rows) + "\n")
print("== adapt\n" + open(f"{OUT}/adapt.tex").read())

# Tab.3 (abl_grouped): ReTrack-style grouped ablation, GAN loss, DiT-XL/2 DPM-50, 10k samples vs split B.
ABL = [
 ("G[A]: Generated-Sample Distribution Alignment (GSA)", None),
 (r"1 & w/o GSA (EAP only)", "v3/none"),
 (r"2 & GSA on $E(x)$ instead of $\zgen$", "v3/enc_gan0.05"),
 (r"3 & GSA with MMD", "v3/gen_mmd1"),
 (r"4 & GSA with FD", "v3/gen_fd10"),
 ("G[B]: Encoder Anchor Preservation (EAP)", None),
 (r"5 & w/o EAP", "v3/gen_gan0.05_noanc"),
 (r"6 & w/o EAP, lower lr ($3{\times}10^{-6}$)", "v3/gen_gan0.05_noanc_lr3e-6"),
 (r"7 & w/o EAP, early stop (500 steps)", "v3/gen_gan0.05_noanc_500"),
]
_ab = [(r"0 & $D_0$", "v3/orig")] + ABL + [(r"\rowcolor{oursbg} & \method (GSA + EAP)", "v3/gen_gan0.05")]
_av = {n: [from_run(n, m) for m, _, _ in AC] for _, n in _ab if n}
_best = [(min if low else max)(v[j] for v in _av.values()) for j, (_, _, low) in enumerate(AC)]
rows = []
for lab, name in _ab:
    if name is None:
        rows += [r"\midrule", r"\multicolumn{6}{l}{\textit{" + lab + r"}} \\"]; continue
    if "rowcolor" in lab:
        rows.append(r"\midrule")
    rows.append(lab + " & " + " & ".join((r"\textbf{%s}" if abs(x - _best[j]) < 1e-9 else "%s") % f.format(x) for j, (x, (m, f, _)) in enumerate(zip(_av[name], AC))) + r" \\")
open(f"{OUT}/abl_grouped.tex", "w").write("\n".join(rows) + "\n")
print("== abl_grouped\n" + open(f"{OUT}/abl_grouped.tex").read())

# Tab. dg_cmp: generator-side (DG) vs decoder-side (SAD) correction on official DiT-XL/2 at the same cfg 1.5
# (seeds 1900-1907 shared by all rows; DG uses its released discriminator, trained on ImageNet train latents).
DGR = [(r"$D_0$", "--", "--", "ditoff_cfg1.5__orig", False),
       (r"DG~\cite{kim2023dg}", r"$+$20\%", "1.28M", "dgcfg1.5__orig", False),
       (r"\method (ours)", "0", "5k", "ditoff_cfg1.5__d40_gen_gan0.05", True),
       (r"DG + \method (ours)", r"$+$20\%", "1.28M", "dgcfg1.5__sad", True)]
_dv = {b: [_tf(b, k) for k in ("FID", "sFID", "Inception Score")] for *_, b, _ in DGR}
_db = [min(v[0] for v in _dv.values()), min(v[1] for v in _dv.values()), max(v[2] for v in _dv.values())]
_d2 = [sorted(v[0] for v in _dv.values())[1], sorted(v[1] for v in _dv.values())[1], sorted((v[2] for v in _dv.values()), reverse=True)[1]]
rows = []
for lab, cost, real, b, ours in DGR:
    cells = [(r"\textbf{%s}" if abs(x - _db[j]) < 1e-9 else r"\underline{%s}" if abs(x - _d2[j]) < 1e-9 else "%s") % f.format(x) for j, (x, f) in enumerate(zip(_dv[b], ("{:.2f}", "{:.2f}", "{:.1f}")))]
    rows.append((r"\rowcolor{oursbg} " if ours else "") + f"{lab} & {cost} & {real} & " + " & ".join(cells) + r" \\")
open(f"{OUT}/dg_cmp.tex", "w").write("\n".join(rows) + "\n")
print("== dg_cmp\n" + open(f"{OUT}/dg_cmp.tex").read())

# Tab. t2i_main: main-text subset of coco_lam (PixArt-alpha on COCO).
_keep = ("$D_0$ &", r"GAN, $\lambda{=}0.01$ &", r"GAN, $\lambda{=}0.05$ &", r"MMD, $\lambda{=}0.1$ &", r"MMD, $\lambda{=}0.3$ &")
# Last column: PickScore win rate vs D_0 (paired by latent); real COCO val2017 images paired with the D_0 sample of the same caption.
_pkey = {"$D_0$ &": "orig", r"GAN, $\lambda{=}0.01$ &": "g01", r"GAN, $\lambda{=}0.05$ &": "gan_cc", r"MMD, $\lambda{=}0.1$ &": "m01", r"MMD, $\lambda{=}0.3$ &": "m03"}
_lines = []
for l in open(f"{OUT}/coco_lam.tex").read().splitlines():
    k = next((k for k in _keep if l.startswith(k)), None)
    if k is None:
        continue
    cells = l.rstrip(" \\").split(" & ")
    if k != "$D_0$ &":
        cells[-1] = f"{100 * _pk[_pkey[k]]['win_rate_vs_D0']:.0f}\\%"
    _lines.append((r"\rowcolor{oursbg} " if k != "$D_0$ &" else "") + " & ".join(cells) + r" \\")
_lines += [r"\midrule", r"Real images & -- & -- & " + f"{_m['coco']['real']['clip_score']:.2f}" + r" & -- & -- & " + f"{100 * _m['coco']['real']['pick_win']:.0f}\\%" + r" \\"]
open(f"{OUT}/t2i_main.tex", "w").write("\n".join(_lines) + "\n")
print("== t2i_main\n" + open(f"{OUT}/t2i_main.tex").read())

# Appendix: decoder-side baselines with the official DiT recipe (250-step DDPM, cfg 1.5, FID-50K, ADM evaluator;
# FD_CLIP vs split B). Baselines trained on the same 40k latent pool / settings as in Tab.3 (r4 runs, run_r4.sh).
OFF = [  # label, TF batch, clip json stem, ours
 (r"Original decoder $D_0$", "ditoff_cfg1.5__orig", "ditoff_orig", False),
 (r"Noise-robust decoder~\cite{zheng2025rae}", "ditoff_cfg1.5__r4_noise0.1", "ditoff_r4_noise0.1", False),
 (r"Paired post-training~\cite{qiu2025robustok,fang2026gar}", "ditoff_cfg1.5__r4_robustok", "ditoff_r4_robustok", False),
 (r"FDDA~\cite{fang2026fdda}$^\ast$", "ditoff_cfg1.5__r4_fdda_2k", "ditoff_r4_fdda_2k", False),
 (r"FDDA~\cite{fang2026fdda}$^\ast$ + EAP", "ditoff_cfg1.5__r4_fdda_2k_anc", "ditoff_r4_fdda_2k_anc", False),
 (r"\method-GAN (ours)", "ditoff_cfg1.5__d40_gen_gan0.05", "ditoff_d40_gen_gan0.05", True),
 (r"\method-MMD (ours)", "ditoff_cfg1.5__d40_gen_mmd1", "ditoff_d40_gen_mmd1", True),
]
def _oclip(stem):
    return json.load(open(f"results/fid50k_b/{stem}.json"))["ditoff_cfg1.5"]["fd_clip"]
_K3 = _KEYS[:3]
OK = _K3 + [("FD_CLIP", "{:.1f}", True)]
ov = [[_tfo(b, k) for k, _, _ in _K3] + [_oclip(c)] for _, b, c, _ in OFF]
rows = []
for j, (_, f, low) in enumerate(OK):
    col = sorted({float(f.format(r[j])) for r in ov}, reverse=not low)
    for r in ov:
        r[j] = (r[j], col[0], col[1] if len(col) > 1 else None)
for i, (lab, _, _, o) in enumerate(OFF):
    cells = []
    for j, (_, f, _) in enumerate(OK):
        x, b1, b2 = ov[i][j]; t = f.format(x); v = float(t)
        t = r"\textbf{" + t + "}" if v == b1 else (r"\underline{" + t + "}" if v == b2 else t)
        cells.append(t)
    if lab.startswith(r"\method-GAN"):
        rows.append(r"\midrule")
    rows.append((r"\rowcolor{oursbg} " if o else "") + lab + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/adapt_official.tex", "w").write("\n".join(rows) + "\n")
print("== adapt_official\n" + open(f"{OUT}/adapt_official.tex").read())
