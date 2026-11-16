"""Pixel-space refiner baseline: a small residual CNN applied AFTER the frozen original decoder,
trained with the same losses as SAD (so 'why the decoder?' can be answered).

Also provides maybe_load_refiner(vae, ckpt) -> decode function used by all evals:
  ckpt == ""                 -> original decoder
  ckpt has decoder.* keys     -> fine-tuned decoder
  ckpt has refiner.* keys     -> original decoder + refiner
"""

import torch
import torch.nn as nn

from vae_utils import decode


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                               nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.f(x)


class Refiner(nn.Module):
    """3 -> 64 (256px) -> 128 (128px) -> 256 (64px, 4 res blocks) -> back up; zero-init residual output."""

    def __init__(self, ch=(64, 128, 256), n_mid=4):
        super().__init__()
        a, b, c = ch
        self.inp = nn.Conv2d(3, a, 3, padding=1)
        self.d1 = nn.Sequential(ResBlock(a), nn.Conv2d(a, b, 4, stride=2, padding=1))
        self.d2 = nn.Sequential(ResBlock(b), nn.Conv2d(b, c, 4, stride=2, padding=1))
        self.mid = nn.Sequential(*[ResBlock(c) for _ in range(n_mid)])
        self.u2 = nn.Sequential(nn.ConvTranspose2d(c, b, 4, stride=2, padding=1), ResBlock(b))
        self.u1 = nn.Sequential(nn.ConvTranspose2d(b, a, 4, stride=2, padding=1), ResBlock(a))
        self.out = nn.Sequential(nn.GroupNorm(32, a), nn.SiLU(), nn.Conv2d(a, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        h0 = self.inp(x)
        h1 = self.d1(h0)
        h2 = self.mid(self.d2(h1))
        h = self.u1(self.u2(h2) + h1) + h0
        return x + self.out(h)


def maybe_load_refiner(vae, ckpt):
    if not ckpt:
        return lambda z: decode(vae, z)
    sd = torch.load(ckpt)
    if any(k.startswith("refiner.") for k in sd):
        a = sd["refiner.inp.weight"].shape[0]
        ref = Refiner((a, 2 * a, 4 * a)).cuda().eval()
        ref.load_state_dict({k[len("refiner."):]: v for k, v in sd.items()})

        @torch.no_grad()
        def dec(z):
            x = decode(vae, z)
            return torch.cat([ref(x[i:i + 50]).clamp(-1, 1) for i in range(0, len(x), 50)])
        return dec
    missing = vae.load_state_dict(sd, strict=False)
    assert not [k for k in missing.missing_keys if k.startswith("decoder.")]
    return lambda z: decode(vae, z)
