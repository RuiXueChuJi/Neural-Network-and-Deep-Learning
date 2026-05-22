"""
common.py — shared helpers used by all experiment scripts.

Provides:
    - load_mnist()         : returns train/dev/test splits as numpy arrays.
    - dump_history(runner) : pull per-epoch and per-iteration metrics out of
                             a runner into a json-friendly dict.
    - save_results(name, history)
    - rotate_aug / shift_aug / noise_aug : data augmentations
"""

import json
import os
import gzip
from struct import unpack

import numpy as np


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _read_idx(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32)
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return imgs, labs


def load_mnist(seed=309, dev_size=10000, root=None):
    if root is None:
        root = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'MNIST')
    train_imgs, train_labs = _read_idx(
        os.path.join(root, 'train-images-idx3-ubyte.gz'),
        os.path.join(root, 'train-labels-idx1-ubyte.gz'),
    )
    test_imgs, test_labs = _read_idx(
        os.path.join(root, 't10k-images-idx3-ubyte.gz'),
        os.path.join(root, 't10k-labels-idx1-ubyte.gz'),
    )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_labs))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    dev_imgs = train_imgs[:dev_size]
    dev_labs = train_labs[:dev_size]
    train_imgs = train_imgs[dev_size:]
    train_labs = train_labs[dev_size:]

    train_imgs = train_imgs / 255.0
    dev_imgs = dev_imgs / 255.0
    test_imgs = test_imgs / 255.0
    return ((train_imgs, train_labs),
            (dev_imgs, dev_labs),
            (test_imgs, test_labs))


# ----------------------------------------------------------------------------
# Metrics dumping
# ----------------------------------------------------------------------------

def dump_history(runner):
    return {
        'iter_loss':         [float(x) for x in runner.train_loss],
        'iter_train_acc':    [float(x) for x in runner.train_scores],
        'epoch_train_loss':  [float(x) for x in runner.epoch_train_loss],
        'epoch_train_acc':   [float(x) for x in runner.epoch_train_score],
        'epoch_dev_loss':    [float(x) for x in runner.epoch_dev_loss],
        'epoch_dev_acc':     [float(x) for x in runner.epoch_dev_score],
        'lr_history':        [float(x) for x in runner.lr_history],
        'best_dev_acc':      float(runner.best_score),
    }


def save_results(name, history, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(history, f)
    return path


# ----------------------------------------------------------------------------
# Test-set evaluation
# ----------------------------------------------------------------------------

def eval_on_test(model, test_set, batch_size=512):
    if hasattr(model, 'eval'):
        model.eval()
    X, y = test_set
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size]))
    logits = np.concatenate(out, axis=0)
    pred = np.argmax(logits, axis=-1)
    return float((pred == y).mean()), logits, pred


# ----------------------------------------------------------------------------
# Data augmentation
# ----------------------------------------------------------------------------

def _rotate_image_28(img28, angle_deg):
    """Cheap nearest-neighbor rotation around the 28x28 image center."""
    H = W = 28
    theta = np.deg2rad(angle_deg)
    cos = np.cos(theta); sin = np.sin(theta)
    yy, xx = np.indices((H, W))
    cy = cx = (H - 1) / 2.0
    src_y = cos * (yy - cy) + sin * (xx - cx) + cy
    src_x = -sin * (yy - cy) + cos * (xx - cx) + cx
    src_y = np.clip(np.round(src_y), 0, H - 1).astype(np.int64)
    src_x = np.clip(np.round(src_x), 0, W - 1).astype(np.int64)
    return img28[src_y, src_x]


def make_augmentor(rotate_max=0.0, shift_max=0, noise_std=0.0,
                   prob_rotate=0.5, prob_shift=0.5, prob_noise=0.5,
                   seed=None, in_shape=(28, 28)):
    """Return an `augmentor(X, y) -> (X', y)` function.

    Works for X of shape [B, 784] (MLP) or [B, 1, 28, 28] (CNN).
    """
    rng = np.random.default_rng(seed)

    def aug(X, y):
        flat = (X.ndim == 2)
        if flat:
            x = X.reshape(-1, *in_shape)
        else:
            x = X.reshape(-1, *in_shape) if X.shape[1] == 1 else X[:, 0]
        x = x.copy()
        B = x.shape[0]

        for i in range(B):
            if rotate_max > 0 and rng.random() < prob_rotate:
                ang = rng.uniform(-rotate_max, rotate_max)
                x[i] = _rotate_image_28(x[i], ang)
            if shift_max > 0 and rng.random() < prob_shift:
                dy = rng.integers(-shift_max, shift_max + 1)
                dx = rng.integers(-shift_max, shift_max + 1)
                x[i] = np.roll(x[i], shift=(dy, dx), axis=(0, 1))
            if noise_std > 0 and rng.random() < prob_noise:
                x[i] = np.clip(x[i] + rng.normal(0, noise_std, size=x[i].shape), 0, 1)

        if flat:
            return x.reshape(B, -1).astype(np.float32), y
        return x.reshape(B, 1, *in_shape).astype(np.float32), y

    return aug
