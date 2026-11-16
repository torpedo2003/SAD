"""Human-preference proxy: PickScore [Kirstain et al. 2023] of decoded samples, and the pairwise win rate of each
decoder against D_0 on the *same* latent and prompt (an automatic stand-in for a 2AFC study).

usage: python pick_eval.py coco|imagenet name:ckpt [name:ckpt ...]     (empty ckpt = D_0, use the name 'orig')
COCO prompts: the captions used for generation; ImageNet prompts: "a photo of a {class name}".
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from refiner import maybe_load_refiner
from vae_utils import load_vae, to_uint8

PK = "pretrained/hf/pickscore"


def prompts(domain):
    if domain == "coco":
        fs = sorted(glob.glob("data/gen/coco_eval_r*.npz"))
        caps = json.load(open("data/coco/eval_caps.json"))
        idx = np.concatenate([np.load(f)["y"] for f in fs])
        return fs, [caps[i] for i in idx]
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    fs = sorted(glob.glob("data/gen/all_cfg1.5_r*.npz"))
    y = np.concatenate([np.load(f)["y"] for f in fs])
    return fs, [f"a photo of a {info.index_to_description(int(c)).split(',')[0]}" for c in y]


@torch.no_grad()
def main():
    domain = sys.argv[1]
    specs = [s.split(":", 1) for s in sys.argv[2:]]
    fs, texts = prompts(domain)
    z = torch.from_numpy(np.concatenate([np.load(f)["z"] for f in fs])).float()
    model = AutoModel.from_pretrained(PK).cuda().eval().half()
    proc = AutoProcessor.from_pretrained(PK)
    tfeat = []
    for i in range(0, len(texts), 256):
        t = proc(text=texts[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        tfeat.append(torch.nn.functional.normalize(model.get_text_features(**t).float(), dim=-1))
    tfeat = torch.cat(tfeat)
    scale = model.logit_scale.exp().float()
    out_p = f"results/pickscore_{domain}.json"
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    scores = {}
    for name, ck in specs:
        sp = f"results/pickscore_{domain}_{name}.npy"
        if os.path.exists(sp):
            scores[name] = np.load(sp)
            continue
        vae = load_vae()
        dec = maybe_load_refiner(vae, ck)
        s = []
        for i in range(0, len(z), 100):
            imgs = to_uint8(dec(z[i:i + 100]))
            px = proc(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
            f = torch.nn.functional.normalize(model.get_image_features(pixel_values=px).float(), dim=-1)
            s.append((scale * (f * tfeat[i:i + 100]).sum(-1)).cpu().numpy())
        scores[name] = np.concatenate(s)
        np.save(sp, scores[name])
        del vae
        torch.cuda.empty_cache()
    base = scores.get("orig")
    for name, v in scores.items():
        r = {"pickscore": float(v.mean())}
        if base is not None and name != "orig":
            r["win_rate_vs_D0"] = float((v > base).mean())
            r["n"] = int(len(v))
        res[name] = r
        print(domain, name, r, flush=True)
    json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
