"""
weight_visualization.py — visualize the first-layer Linear weights of an MLP
or the conv kernels of a CNN.

Defaults to the final MLP saved by experiments/train_final.py.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import mynn as nn


def viz_mlp(path, n_show=64, save_path='figs/mlp_weights.png'):
    m = nn.models.Model_MLP()
    m.load_model(path)
    W = None
    for layer in m.layers:
        if layer.__class__.__name__ == 'Linear':
            W = layer.params['W']
            break
    assert W is not None and W.shape[0] == 784, 'first layer is not 784->H'

    n_show = min(n_show, W.shape[1])
    grid = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.3, grid * 1.3))
    fig.suptitle('MLP first-layer weight images', fontsize=14, fontweight='bold')
    for k in range(grid * grid):
        r, c = k // grid, k % grid
        ax = axes[r, c]
        if k < n_show:
            w = W[:, k].reshape(28, 28)
            vmax = np.max(np.abs(w))
            ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'wrote {save_path}')


def viz_cnn(path, save_path='figs/cnn_kernels.png'):
    m = nn.models.Model_CNN()
    m.load_model(path)
    convs = [l for l in m.layers if l.__class__.__name__ == 'conv2D']
    assert convs, 'no conv layers'

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle('CNN conv kernels (top: layer 1, bottom: layer 2 mean over input ch)',
                 fontsize=13, fontweight='bold')
    W = convs[0].params['W']  # [out, in, k, k]
    for k in range(min(8, W.shape[0])):
        kern = W[k, 0]
        v = np.max(np.abs(kern))
        axes[0, k].imshow(kern, cmap='RdBu_r', vmin=-v, vmax=v)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
    if len(convs) >= 2:
        W2 = convs[1].params['W']
        for k in range(min(8, W2.shape[0])):
            kern = W2[k].mean(0)
            v = np.max(np.abs(kern))
            axes[1, k].imshow(kern, cmap='RdBu_r', vmin=-v, vmax=v)
            axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    else:
        for k in range(8):
            axes[1, k].axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'wrote {save_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.path.join('saved_models', 'final_mlp.pickle'))
    p.add_argument('--kind', default='mlp', choices=['mlp', 'cnn'])
    p.add_argument('--out', default=None)
    args = p.parse_args()

    if args.kind == 'mlp':
        viz_mlp(args.model, save_path=args.out or 'figs/mlp_weights.png')
    else:
        viz_cnn(args.model, save_path=args.out or 'figs/cnn_kernels.png')


if __name__ == '__main__':
    main()
