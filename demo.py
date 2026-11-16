"""Side-by-side demo: the same DiT-XL/2 latents decoded by the original decoder (D_0) and by SAD.

    python demo.py --classes 279 963 388 417 --out assets/demo.png

The generator, the sampler and the latents are identical; only the decoder weights are swapped.
"""
import argparse
import copy

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image

from sad import load_sad_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="+", default=[279, 963, 388, 417])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="dit-xl2-256-dpm50/sad-gan", help="SAD checkpoint name (see sad/hub.py)")
    ap.add_argument("--local_path", default=None, help="load the checkpoint from a local .safetensors file")
    ap.add_argument("--out", default="assets/demo.png")
    a = ap.parse_args()

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 1) sample once with the original decoder and keep the latents it decodes
    captured = {}
    decode_d0 = pipe.vae.decode

    def capture(z, *args, **kwargs):
        captured["z"] = z
        return decode_d0(z, *args, **kwargs)

    pipe.vae.decode = capture
    g = torch.Generator("cuda").manual_seed(a.seed)
    pipe(class_labels=a.classes, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g, output_type="np")
    pipe.vae.decode = decode_d0
    z = captured["z"]  # already divided by the VAE scaling factor

    # 2) decode the SAME latents with D_0 and with SAD
    vae_d0 = pipe.vae
    vae_sad = load_sad_decoder(copy.deepcopy(vae_d0).float(), a.ckpt, local_path=a.local_path).half()

    @torch.no_grad()
    def decode(vae):
        x = vae.decode(z).sample
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    rows = [decode(vae_d0), decode(vae_sad)]
    n, s = len(a.classes), 256
    canvas = Image.new("RGB", (n * s, 2 * s), "white")
    for r, imgs in enumerate(rows):
        for i, im in enumerate(imgs):
            canvas.paste(Image.fromarray(im), (i * s, r * s))
    canvas.save(a.out)
    print(f"saved {a.out}: top row D_0, bottom row SAD")


if __name__ == "__main__":
    main()
