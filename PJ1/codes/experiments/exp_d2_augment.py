"""
Direction 2 — Experiment 2.3: Data augmentation comparison.

Compares:
    - none
    - random rotation [-10°, +10°]
    - random shift +/- 2 pixels
    - small Gaussian noise
    - rotate + shift + noise (combined)
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import (load_mnist, dump_history, save_results,
                                eval_on_test, make_augmentor)


SIZE_LIST = [784, 512, 10]


def run_one(name, augmentor, epochs, batch_size, lr, seed=309):
    np.random.seed(seed)
    train, dev, test = load_mnist(seed=seed)

    model = nn.models.Model_MLP(SIZE_LIST, 'ReLU', init='he')
    optimizer = nn.optimizer.Adam(lr, model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size, augmentor=augmentor)
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'best_models', f'd2_aug_{name}')
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'best_aug_{name}.pickle')
    model.load_model(os.path.join(save_dir, f'best_aug_{name}.pickle'))
    test_acc, *_ = eval_on_test(model, test)

    h = dump_history(runner)
    h['name'] = name
    h['test_acc'] = test_acc
    save_results(f'd2_aug_{name}', h)
    print(f'  -> [{name}] dev={runner.best_score:.4f} test={test_acc:.4f}')
    return runner.best_score, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    augmentors = {
        'none':      None,
        'rotate':    make_augmentor(rotate_max=10, prob_rotate=0.7, seed=1),
        'shift':     make_augmentor(shift_max=2, prob_shift=0.7, seed=2),
        'noise':     make_augmentor(noise_std=0.08, prob_noise=0.7, seed=3),
        'combined':  make_augmentor(rotate_max=10, shift_max=2, noise_std=0.05,
                                    prob_rotate=0.5, prob_shift=0.5, prob_noise=0.5, seed=4),
    }

    summary = {}
    for name, aug in augmentors.items():
        print(f'\n=== Augmentation: {name} ===')
        dev, test = run_one(name, aug, args.epochs, args.batch_size, args.lr)
        summary[name] = {'dev': dev, 'test': test}

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  {k:>10}  dev={v["dev"]:.4f}  test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
