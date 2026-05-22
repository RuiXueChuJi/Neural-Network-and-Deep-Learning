"""
test_model.py — load a trained model and report its accuracy on the test set.

Usage (defaults to the best MLP):
    python test_model.py
    python test_model.py --model saved_models/best_mlp.pickle
    python test_model.py --model saved_models/best_cnn.pickle --kind cnn
"""

import argparse
import gzip
import os
from struct import unpack

import numpy as np

import mynn as nn


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32)
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return imgs, labs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.path.join('saved_models', 'best_mlp.pickle'))
    p.add_argument('--kind', default='mlp', choices=['mlp', 'cnn'])
    p.add_argument('--batch_size', type=int, default=512)
    args = p.parse_args()

    test_images_path = os.path.join('dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join('dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

    test_imgs, test_labs = load_mnist(test_images_path, test_labels_path)
    test_imgs = test_imgs / 255.0

    if args.kind == 'mlp':
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model.load_model(args.model)
    if hasattr(model, 'eval'):
        model.eval()

    # batched forward
    out = []
    for i in range(0, test_imgs.shape[0], args.batch_size):
        out.append(model(test_imgs[i:i + args.batch_size]))
    logits = np.concatenate(out, axis=0)

    acc = nn.metric.accuracy(logits, test_labs)
    print(f'Test accuracy of {args.model} = {acc * 100:.2f}%')


if __name__ == '__main__':
    main()
