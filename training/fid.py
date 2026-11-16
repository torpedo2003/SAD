"""Inception (pytorch-fid weights) features and Frechet distance helpers."""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3
from scipy import linalg

_model = None


def inception():
    global _model
    if _model is None:
        _model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    return _model


@torch.no_grad()
def features(images, batch=250):
    """images: uint8 [N,H,W,3] numpy or float tensor in [0,1] [N,3,H,W]."""
    model = inception()
    feats = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
        x = x.cuda().float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        feats.append(model(x)[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(feats).numpy()


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet(s1, s2):
    mu1, c1 = s1
    mu2, c2 = s2
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(c1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((c1 + off) @ (c2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.trace(covmean))
