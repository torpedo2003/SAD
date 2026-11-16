"""Sample-Aligned Decoders (SAD): drop-in decoders for latent generative models.

    from diffusers import AutoencoderKL
    from sad import load_sad_decoder

    vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae")
    load_sad_decoder(vae, "dit-xl2-256/sad-gan")      # only the decoder weights change

Everything else in your sampling pipeline stays the same.
"""
from .hub import HF_REPO, CHECKPOINTS, load_sad_decoder

__all__ = ["HF_REPO", "CHECKPOINTS", "load_sad_decoder"]
