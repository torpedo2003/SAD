"""VA-VAE (f16d32) with the same interface as the diffusers AutoencoderKL used elsewhere, so that train_sad.py and the
evaluation scripts work unchanged when SAD_VAE=vavae. Latents are in raw VAE space, as for the SD-VAE."""
import sys
from types import SimpleNamespace

sys.path.insert(0, "third_party/LightningDiT")
from tokenizer.autoencoder import AutoencoderKL as _Base  # noqa: E402

CKPT = "pretrained/lightningdit/vavae-imagenet256-f16d32-dinov2.pt"


class VAVAE(_Base):
    def encode(self, x):
        return SimpleNamespace(latent_dist=super().encode(x))

    def decode(self, z):
        return SimpleNamespace(sample=super().decode(z))

    def enable_gradient_checkpointing(self):  # the f16 decoder is small enough without it
        pass


E2E = "pretrained/repae/e2e-vavae-400k.pt"


def load_vavae(e2e=False):
    if not e2e:
        return VAVAE(embed_dim=32, ch_mult=(1, 1, 2, 2, 4), ckpt_path=CKPT).cuda().eval()
    # REPA-E's end-to-end tuned VA-VAE (same architecture and parameter names)
    m = VAVAE(embed_dim=32, ch_mult=(1, 1, 2, 2, 4))
    m.load_state_dict(__import__("torch").load(E2E, map_location="cpu"))
    return m.cuda().eval()
