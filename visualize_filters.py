"""Visualize the first convolutional layer filters of a trained CIFAR-10 model.

The first conv weight has shape (out_channels, 3, kH, kW). With 3 input channels
we can render each filter as an RGB patch, normalized to [0,1] per filter, and
tile them into a grid -- the classic "what did layer-1 learn" picture
(edges / color-opponent / blob detectors).

Usage:
    python visualize_filters.py --ckpt reports/task1/bn/best_model.pt \
        --out reports/task1/bn/first_layer_filters.png
"""
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_first_conv(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    for k, v in sd.items():
        if hasattr(v, "dim") and v.dim() == 4:
            return k, v.detach().cpu().numpy()
    raise RuntimeError("no 4D conv weight found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    name, w = load_first_conv(args.ckpt)  # (C_out, 3, kH, kW)
    n = w.shape[0]
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            f = w[i].transpose(1, 2, 0)  # (kH, kW, 3)
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)  # per-filter min-max
            ax.imshow(f, interpolation="nearest")
    fig.suptitle(f"First-layer conv filters: {name}  ({n} x 3 x {w.shape[2]} x {w.shape[3]})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("saved", args.out, "| filters", n, "| shape", tuple(w.shape))


if __name__ == "__main__":
    main()
