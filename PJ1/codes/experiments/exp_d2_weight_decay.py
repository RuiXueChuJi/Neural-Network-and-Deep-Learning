"""
Direction 2 — Experiment 2.1: L2 weight decay sweep.

We compare several weight-decay values (lambda) using AdamW so the
weight-decay term is decoupled from the gradient.
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import load_mnist, dump_history, save_results, eval_on_test


SIZE_LIST = [784, 512, 10]


def run_one(lam, epochs, batch_size, lr, seed=309):
    np.random.seed(seed)
    train, dev, test = load_mnist(seed=seed)

    if lam is None or lam == 0:
        lambda_list = None
    else:
        lambda_list = [lam] * (len(SIZE_LIST) - 1)

    model = nn.models.Model_MLP(SIZE_LIST, 'ReLU', lambda_list=lambda_list, init='he')
    optimizer = nn.optimizer.AdamW(lr, model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size)
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'best_models', f'd2_wd_{lam}')
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'best_wd_{lam}.pickle')
    model.load_model(os.path.join(save_dir, f'best_wd_{lam}.pickle'))
    test_acc, *_ = eval_on_test(model, test)

    h = dump_history(runner)
    h['lambda'] = lam
    h['test_acc'] = test_acc
    save_results(f'd2_wd_{lam}', h)
    print(f'  -> [lambda={lam}] dev={runner.best_score:.4f} test={test_acc:.4f}')
    return runner.best_score, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    summary = {}
    for lam in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
        print(f'\n=== Weight-decay lambda = {lam} ===')
        dev, test = run_one(lam, args.epochs, args.batch_size, args.lr)
        summary[lam] = {'dev': dev, 'test': test}

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  lambda={k:<8}  dev={v["dev"]:.4f}  test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
