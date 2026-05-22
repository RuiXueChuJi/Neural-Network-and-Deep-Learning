"""
test_train.py — train an MLP on MNIST with SGD + MultiStepLR.

This is the simple "default" recipe; full experiments live under
`experiments/`.

Usage:
    python test_train.py
"""

import os
import sys
import gzip
import pickle
from struct import unpack

import numpy as np
import matplotlib.pyplot as plt

import mynn as nn
from draw_tools.plot import plot


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32)
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return imgs, labs


def main():
    np.random.seed(309)
    train_images_path = os.path.join('dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join('dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

    train_imgs, train_labs = load_mnist(train_images_path, train_labels_path)

    idx = np.random.permutation(len(train_labs))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0

    # Default recipe: MLP + SGD + MultiStepLR
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4], init='he')
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=128, scheduler=scheduler)
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                 num_epochs=5, log_iters=100, save_dir='./best_models')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.set_tight_layout(True)
    plot(runner, axes)
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/test_train_curve.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
