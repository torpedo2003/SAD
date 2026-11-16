"""Shared VAE helpers. Latents are in raw VAE space (NOT multiplied by scaling_factor)."""

import io
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image


def load_vae():
    if os.environ.get("SAD_VAE") == "vavae":  # second tokenizer: VA-VAE f16d32 (LightningDiT)
        from vavae_utils import load_vavae
        return load_vavae()
    if os.environ.get("SAD_VAE") == "e2evavae":  # third tokenizer: REPA-E's E2E-VAVAE (SiT-XL/1 + REPA on the frozen E2E-VAVAE)
        from vavae_utils import load_vavae
        return load_vavae(e2e=True)
    # DiT-XL-2-256 bundles its VAE (sd-vae-ft-ema); use the same one everywhere.
    return AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae").cuda().eval()


def to_uint8(x):
    """[-1,1] float [N,3,H,W] -> uint8 numpy [N,H,W,3]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def from_uint8(im):
    """uint8 numpy [N,H,W,3] -> [-1,1] float cuda [N,3,H,W]."""
    return torch.from_numpy(im).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1


@torch.no_grad()
def decode(vae, z, batch=50):
    """Returns float images in [-1,1] (clamped), on cuda."""
    out = []
    for i in range(0, len(z), batch):
        out.append(vae.decode(z[i:i + batch].cuda().float()).sample.clamp(-1, 1))
    return torch.cat(out)


@torch.no_grad()
def encode(vae, x, batch=50, gen=None):
    """x: float [-1,1] cuda. Returns (posterior mean, posterior sample) on cpu."""
    means, samples = [], []
    for i in range(0, len(x), batch):
        d = vae.encode(x[i:i + batch].float()).latent_dist
        means.append(d.mean.cpu())
        samples.append((d.mean + d.std * torch.randn(d.mean.shape, generator=gen, device="cuda")).cpu())
    return torch.cat(means), torch.cat(samples)


def center_crop_256(img):
    """ADM-style center crop + resize to 256 (no-op if already 256x256)."""
    img = img.convert("RGB")
    if img.size == (256, 256):
        return np.asarray(img)
    while min(img.size) >= 512:
        img = img.resize(tuple(s // 2 for s in img.size), Image.BOX)
    scale = 256 / min(img.size)
    img = img.resize(tuple(round(s * scale) for s in img.size), Image.BICUBIC)
    a = np.asarray(img)
    y, x = (a.shape[0] - 256) // 2, (a.shape[1] - 256) // 2
    return a[y:y + 256, x:x + 256]


def decode_jpeg(b):
    return center_crop_256(Image.open(io.BytesIO(b)))
