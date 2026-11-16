"""|SAD - D_0| heat map of the zoomed crops (inferno-like colormap, per-image 99.5th-percentile scaling).

usage: python analysis/make_qual_diff.py idx1,idx2,... [full]
  reads  paper/fig/assets/qual/sel_{idx}_{orig,sad}_crop.png (or sel_{idx}_{orig,sad}.png with "full")
  writes paper/fig/assets/qual/sel_{idx}_diff_crop.png (or sel_{idx}_diff.png)
"""
import sys

import numpy as np
from PIL import Image

INF = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21), (250, 193, 39),
       (252, 255, 164)]


def cmap(x, anchors=INF):
    a, t = np.array(anchors, np.float32), np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(x, t, a[:, c]) for c in range(3)], -1).astype(np.uint8)


def main():
  suf = "" if len(sys.argv) > 2 and sys.argv[2] == "full" else "_crop"
  for i in sys.argv[1].split(","):
    L = lambda n: np.asarray(Image.open(f"paper/fig/assets/qual/sel_{i}_{n}{suf}.png").convert("RGB"), np.float32)
    d = np.abs(L("sad") - L("orig")).mean(-1)
    Image.fromarray(cmap(np.clip(d / np.percentile(d, 99.5), 0, 1))).save(f"paper/fig/assets/qual/sel_{i}_diff{suf}.png")


if __name__ == "__main__":
    main()
