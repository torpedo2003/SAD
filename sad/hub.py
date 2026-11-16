import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HF_REPO = "torpedo2003/SAD"

# name -> (file in the Hugging Face repo, tokenizer the decoder belongs to)
CHECKPOINTS = {
    "dit-xl2-256/sad-gan": ("dit-xl2-256/sad-gan.safetensors", "SD-VAE (sd-vae-ft-ema), official DiT 250-step DDPM sampler"),
    "dit-xl2-256/sad-mmd": ("dit-xl2-256/sad-mmd.safetensors", "SD-VAE (sd-vae-ft-ema), official DiT 250-step DDPM sampler"),
    "dit-xl2-256-dpm50/sad-gan": ("dit-xl2-256-dpm50/sad-gan.safetensors", "SD-VAE (sd-vae-ft-ema), 50-step DPM-Solver++"),
    "dit-xl2-256-dpm50/sad-mmd": ("dit-xl2-256-dpm50/sad-mmd.safetensors", "SD-VAE (sd-vae-ft-ema), 50-step DPM-Solver++"),
    "sit-xl2-repa-256/sad-gan": ("sit-xl2-repa-256/sad-gan.safetensors", "SD-VAE (sd-vae-ft-ema), SiT-XL/2 + REPA, SDE 250 steps"),
    "lightningdit-xl-vavae-256/sad-gan": ("lightningdit-xl-vavae-256/sad-gan.safetensors", "VA-VAE (f16d32), LightningDiT-XL"),
    "repa-e-e2evae-256/sad-gan": ("repa-e-e2evae-256/sad-gan.safetensors", "E2E-VAE, REPA-E SiT-XL/1, SDE 250 steps"),
    "pixart-alpha-256/sad-gan": ("pixart-alpha-256/sad-gan.safetensors", "SD-VAE (sd-vae-ft-ema), PixArt-alpha 256px"),
}


def load_sad_decoder(vae: torch.nn.Module, name: str = None, repo: str = HF_REPO, local_path: str = None) -> torch.nn.Module:
    """Replace the decoder of `vae` (in place) with a SAD decoder and return `vae`.

    Only `decoder.*` and `post_quant_conv.*` parameters are loaded; the encoder is untouched,
    so the latent space and the generator are exactly the same as before.
    """
    path = local_path or hf_hub_download(repo, CHECKPOINTS[name][0] if name in CHECKPOINTS else name)
    sd = torch.load(path, map_location="cpu") if str(path).endswith(".pt") else load_file(path)  # .pt: training/train_sad.py output
    result = vae.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if k.startswith(("decoder.", "post_quant_conv."))]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint does not match this VAE: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}")
    return vae
