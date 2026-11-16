"""Shared data / feature utilities for G1 decoder fine-tuning."""

import glob
import random

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
import torch.nn.functional as F

from vae_utils import decode_jpeg

SHARDS = "data/imagenet-1k-256x256/train-{:05d}-of-00040.parquet"
# Shards are sorted by class and each row group holds 100 images of ONE class, so the stream must mix
# many row groups. All downloaded shards are used (the 10k real75 eval images are ~0.8% of shards 0-2).
TRAIN_SHARDS = list(range(0, 39))


class RealImages(torch.utils.data.IterableDataset):
    """Infinite stream of uint8 [256,256,3] ImageNet images, shuffled across `mix` random row groups."""

    def __init__(self, shards=TRAIN_SHARDS, seed=0, mix=64):
        self.files = [f for s in shards for f in glob.glob(SHARDS.format(s))]
        self.seed, self.mix = seed, mix

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        pfs = [pq.ParquetFile(f) for f in self.files]
        groups = [(i, g) for i, pf in enumerate(pfs) for g in range(pf.num_row_groups)]
        while True:
            buf = []
            for i, g in rng.sample(groups, self.mix):
                buf.extend(pfs[i].read_row_group(g, columns=["image"])["image"].to_pylist())
            rng.shuffle(buf)
            for b in buf:
                yield torch.from_numpy(decode_jpeg(b["bytes"]).copy())


def real_loader(batch, seed=0, workers=6):
    return iter(torch.utils.data.DataLoader(RealImages(seed=seed), batch_size=batch, num_workers=workers,
                                            pin_memory=True, persistent_workers=True))


class FeatNet(torch.nn.Module):
    """Frozen timm backbone; input [-1,1] images [N,3,H,W]; returns pooled features."""

    def __init__(self, name, size=224):
        super().__init__()
        # DINOv2's default input is 518px; run every backbone at 224 (pos-embed interpolated at load)
        self.m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).cuda().eval().requires_grad_(False)
        cfg = self.m.pretrained_cfg
        self.size = size
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda())

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x.float(), size=(self.size, self.size), mode="bicubic", align_corners=False)
        return self.m((x - self.mean) / self.std).float()


DINO = "vit_small_patch14_dinov2.lvd142m"
CLIP = "vit_base_patch16_clip_224.openai"


def load_gen_train(pools="cfg1.5"):
    # comma-separated pool names under data/gen_train, e.g. "cfg1.5" (DiT) or "sit_cfg1.5" or "cfg1.5,sit_cfg1.5" (mixed)
    fs = sorted(f for p in pools.split(",") for f in glob.glob(f"data/gen_train/{p}_r*.npz"))
    assert fs, pools
    return torch.from_numpy(np.concatenate([np.load(f)["z"] for f in fs]))


def load_gen_train_labels(pools="cfg1.5"):
    fs = sorted(f for p in pools.split(",") for f in glob.glob(f"data/gen_train/{p}_r*.npz"))
    return torch.from_numpy(np.concatenate([np.load(f)["y"] for f in fs]).astype(np.int64))
