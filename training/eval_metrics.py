"""Extra evaluation metrics, on the same latents and real references as the existing protocols.

Distribution metrics: CMMD (CLIP ViT-L/14@336, Gaussian kernel, sigma=10, x1000), KID (Inception, x1000), FD_SigLIP.
Preference metrics:   HPS v2.1 and PickScore, with the pairwise win rate against D_0 on the same latent and prompt.
                      On COCO the real images are scored with their own captions, as a reference level.

usage: python eval_metrics.py coco|imagenet name:ckpt [name:ckpt ...]     (empty ckpt = D_0, use the name 'orig')
output: results/metrics_<domain>.json; per-decoder features and scores are cached in results/metrics/.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import fid
from refiner import maybe_load_refiner
from vae_utils import load_vae, to_uint8

PD = "pretrained/hf"
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)
HPS_CKPT = "pretrained/hps/HPS_v2.1_compressed.pt"


def tensors(u8, size, mean, std):
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float().cuda() / 255
    x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    m, s = torch.tensor(mean).cuda()[None, :, None, None], torch.tensor(std).cuda()[None, :, None, None]
    return ((x - m) / s).half()


def in_batches(u8, fn, bs=64):
    return np.concatenate([fn(u8[i:i + bs]) for i in range(0, len(u8), bs)])


class Clip336:
    key = "clip336"

    def __init__(self):
        from transformers import CLIPVisionModelWithProjection
        self.m = CLIPVisionModelWithProjection.from_pretrained(f"{PD}/clip-vit-large-patch14-336").cuda().eval().half()

    @torch.no_grad()
    def __call__(self, u8):
        e = self.m(pixel_values=tensors(u8, 336, OPENAI_MEAN, OPENAI_STD)).image_embeds
        return F.normalize(e.float(), dim=-1).cpu().numpy()


class Siglip:
    key = "siglip"

    def __init__(self):
        from transformers import SiglipVisionModel
        self.m = SiglipVisionModel.from_pretrained(f"{PD}/siglip-base-patch16-224").cuda().eval().half()

    @torch.no_grad()
    def __call__(self, u8):
        return self.m(pixel_values=tensors(u8, 224, HALF, HALF)).pooler_output.float().cpu().numpy()


def cmmd(x, y, sigma=10.0, bs=2000):
    x, y = torch.from_numpy(x).cuda().double(), torch.from_numpy(y).cuda().double()
    g = 1 / (2 * sigma ** 2)

    def mean_k(a, b):
        t = 0.0
        for i in range(0, len(a), bs):
            t += torch.exp(-g * torch.cdist(a[i:i + bs], b).pow(2)).sum().item()
        return t / (len(a) * len(b))
    return 1000 * (mean_k(x, x) + mean_k(y, y) - 2 * mean_k(x, y))


def kid(x, y, subsets=100, size=1000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = torch.from_numpy(x).cuda().double(), torch.from_numpy(y).cuda().double()
    d, out = x.shape[1], []
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    for _ in range(subsets):
        a, b = x[rng.choice(len(x), size, replace=False)], y[rng.choice(len(y), size, replace=False)]
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        out.append(((kaa.sum() - kaa.trace() + kbb.sum() - kbb.trace()) / (size * (size - 1)) - 2 * kab.mean()).item())
    return 1000 * float(np.mean(out))


class Hps:
    def __init__(self, texts):
        import open_clip
        self.m = open_clip.create_model("ViT-H-14", pretrained=None).cuda().eval()
        sd = torch.load(HPS_CKPT, map_location="cpu", weights_only=True)["state_dict"]
        missing = self.m.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=False).missing_keys
        assert not missing, f"HPS checkpoint does not cover the model: {missing[:5]}"
        self.m.half()
        tok = open_clip.get_tokenizer("ViT-H-14")
        with torch.no_grad():
            self.t = torch.cat([F.normalize(self.m.encode_text(tok(texts[i:i + 256]).cuda()).float(), dim=-1)
                                for i in range(0, len(texts), 256)])

    @torch.no_grad()
    def score(self, u8, bs=64):
        s = []
        for i in range(0, len(u8), bs):
            f = F.normalize(self.m.encode_image(tensors(u8[i:i + bs], 224, OPENAI_MEAN, OPENAI_STD)).float(), dim=-1)
            s.append((f * self.t[i:i + bs]).sum(-1).cpu().numpy())
        return np.concatenate(s)


class Pick:
    def __init__(self, texts):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(f"{PD}/pickscore").cuda().eval().half()
        self.p = AutoProcessor.from_pretrained(f"{PD}/pickscore")
        with torch.no_grad():
            t = []
            for i in range(0, len(texts), 256):
                b = self.p(text=texts[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
                t.append(F.normalize(self.m.get_text_features(**b).float(), dim=-1))
        self.t, self.scale = torch.cat(t), self.m.logit_scale.exp().float()

    @torch.no_grad()
    def score(self, u8, bs=64):
        s = []
        for i in range(0, len(u8), bs):
            px = self.p(images=list(u8[i:i + bs]), return_tensors="pt")["pixel_values"].cuda().half()
            f = F.normalize(self.m.get_image_features(pixel_values=px).float(), dim=-1)
            s.append((self.scale * (f * self.t[i:i + bs]).sum(-1)).cpu().numpy())
        return np.concatenate(s)


def domain_data(domain):
    if domain == "coco":
        fs = sorted(glob.glob("data/gen/coco_eval_r*.npz"))
        caps = json.load(open("data/coco/eval_caps.json"))
        idx = np.concatenate([np.load(f)["y"] for f in fs])
        real = np.load("data/coco/ref.npz")["images"]
        # eval_caps holds two captions per reference image, in reference order
        return fs, [caps[i] for i in idx], real, [caps[2 * i] for i in range(len(real))]
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    # imagenet: DPM-50 DiT 10k; ditoff: official-code DiT (first 2 shards, 12.5k); ldit: LightningDiT 10k (SAD_VAE=vavae)
    # repae: REPA-E SiT-XL/1 on E2E-VAE (first 2 shards, 12.5k; SAD_VAE=e2evavae)
    # sit: SiT-XL/2 Heun-50 10k (SD-VAE); repa: SiT+REPA SDE-250 (first 2 shards, 12.5k; SD-VAE)
    pat = {"imagenet": "data/gen/all_cfg1.5_r*.npz", "ditoff": "data/gen/ditoff_cfg1.5_r[01].npz", "ldit": "data/gen/ldit10k_r*.npz",
           "repae": "data/gen/repae_eval_r[01].npz", "sit": "data/gen/sit_cfg1.5_r*.npz", "repa": "data/gen/repa_eval_r[01].npz"}[domain]
    fs = sorted(glob.glob(pat))
    y = np.concatenate([np.load(f)["y"] for f in fs])
    real = np.load("data/VIRTUAL_imagenet256_labeled.npz")["arr_0"][np.load("data/adm_split.npz")["B"]]
    return fs, [f"a photo of a {info.index_to_description(int(c)).split(',')[0]}" for c in y], real, None


def main():
    domain = sys.argv[1]
    specs = [s.split(":", 1) for s in sys.argv[2:]]
    os.makedirs("results/metrics", exist_ok=True)
    fs, texts, real, real_texts = domain_data(domain)
    z = torch.from_numpy(np.concatenate([np.load(f)["z"] for f in fs])).float()
    nets = [Clip336(), Siglip()]
    feats = lambda u8: {**{n.key: in_batches(u8, n) for n in nets}, "inception": fid.features(u8)}

    rp = f"results/metrics/{domain}_real.npz"
    if not os.path.exists(rp):
        np.savez(rp, **feats(real))
    ref = dict(np.load(rp))

    hps, pick = Hps(texts), Pick(texts)
    res_p = f"results/metrics_{domain}.json"
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    scores = {}
    for name, ck in specs:
        cp = f"results/metrics/{domain}_{name}.npz"
        if not os.path.exists(cp):
            vae = load_vae()
            dec = maybe_load_refiner(vae, ck)
            with torch.no_grad():
                u8 = np.concatenate([to_uint8(dec(z[i:i + 50])) for i in range(0, len(z), 50)])
            del vae, dec
            torch.cuda.empty_cache()
            np.savez(cp, **feats(u8), hps=hps.score(u8), pick=pick.score(u8))
        c = dict(np.load(cp))
        scores[name] = c
        res[name] = {"n": len(z), "cmmd": cmmd(c["clip336"], ref["clip336"]), "kid": kid(c["inception"], ref["inception"]),
                     "fd_siglip": fid.frechet(fid.stats(c["siglip"]), fid.stats(ref["siglip"])),
                     "hps": float(c["hps"].mean()), "pick": float(c["pick"].mean())}
        print(domain, name, res[name], flush=True)
        json.dump(res, open(res_p, "w"), indent=1)
    if "orig" in scores:
        base = scores["orig"]
        for name, c in scores.items():
            if name != "orig":
                res[name].update(hps_win=float((c["hps"] > base["hps"]).mean()), pick_win=float((c["pick"] > base["pick"]).mean()))
                print(domain, name, "win", res[name]["hps_win"], res[name]["pick_win"], flush=True)
    if real_texts is not None and "real" not in res:
        rs = {"hps": Hps(real_texts).score(real), "pick": Pick(real_texts).score(real)}
        res["real"] = {k: float(v.mean()) for k, v in rs.items()}
        print(domain, "real", res["real"], flush=True)
    json.dump(res, open(res_p, "w"), indent=1)


if __name__ == "__main__":
    main()
