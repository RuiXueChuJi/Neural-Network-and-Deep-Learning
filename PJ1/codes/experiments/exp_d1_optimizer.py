"""
Direction 1 — Experiment 1.1: Optimizer comparison.

Compares: SGD, MomentGD (vanilla momentum), MomentGD (Nesterov), Adam, AdamW.
All use the same MLP, batch size, schedule, and number of epochs.
"""

import os
import sys
import argparse
import numpy as np

# allow `python experiments/exp_d1_optimizer.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import load_mnist, dump_history, save_results, eval_on_test


SIZE_LIST = [784, 512, 10]


def build_optimizer(name, model, lr):
    if name == 'sgd':
        return nn.optimizer.SGD(lr, model)
    if name == 'momentum':
        return nn.optimizer.MomentGD(lr, model, mu=0.9, nesterov=False)
    if name == 'nesterov':
        return nn.optimizer.MomentGD(lr, model, mu=0.9, nesterov=True)
    if name == 'adam':
        return nn.optimizer.Adam(lr, model)
    if name == 'adamw':
        return nn.optimizer.AdamW(lr, model)
    raise ValueError(name)


def run_one(name, lr, epochs, batch_size, seed=309):
    np.random.seed(seed)
    (train, dev, test) = load_mnist(seed=seed)

    model = nn.models.Model_MLP(SIZE_LIST, 'ReLU', init='he')
    optimizer = build_optimizer(name, model, lr)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size, scheduler=None, verbose=True)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'best_models', f'd1_optim_{name}')
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'best_{name}.pickle')

    # Evaluate on test
    model.load_model(os.path.join(save_dir, f'best_{name}.pickle'))
    test_acc, *_ = eval_on_test(model, test)

    history = dump_history(runner)
    history['name'] = name
    history['lr'] = lr
    history['test_acc'] = test_acc
    save_results(f'd1_optimizer_{name}', history)
    print(f'  -> [{name}] best dev = {runner.best_score:.4f}, test = {test_acc:.4f}')
    return runner.best_score, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    # Each optimizer gets a sensible lr.
    configs = [
        ('sgd',      0.05),
        ('momentum', 0.05),
        ('nesterov', 0.05),
        ('adam',     1e-3),
        ('adamw',    1e-3),
    ]

    summary = {}
    for name, lr in configs:
        print(f'\n=== Optimizer: {name} (lr={lr}) ===')
        dev, test = run_one(name, lr, args.epochs, args.batch_size)
        summary[name] = {'dev': dev, 'test': test, 'lr': lr}

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  {k:>8}  dev={v["dev"]:.4f}  test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
