"""
Direction 2 — Experiment 2.2: Dropout sweep.
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import load_mnist, dump_history, save_results, eval_on_test


SIZE_LIST = [784, 512, 256, 10]


def run_one(p_drop, epochs, batch_size, lr, seed=309):
    np.random.seed(seed)
    train, dev, test = load_mnist(seed=seed)

    model = nn.models.Model_MLP(SIZE_LIST, 'ReLU', init='he', dropout=p_drop)
    optimizer = nn.optimizer.Adam(lr, model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size)
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'best_models', f'd2_drop_{p_drop}')
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'best_drop_{p_drop}.pickle')
    model.load_model(os.path.join(save_dir, f'best_drop_{p_drop}.pickle'))
    test_acc, *_ = eval_on_test(model, test)

    h = dump_history(runner)
    h['dropout'] = p_drop
    h['test_acc'] = test_acc
    save_results(f'd2_dropout_{p_drop}', h)
    print(f'  -> [p={p_drop}] dev={runner.best_score:.4f} test={test_acc:.4f}')
    return runner.best_score, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    summary = {}
    for pd in [0.0, 0.1, 0.2, 0.3, 0.5]:
        print(f'\n=== Dropout p = {pd} ===')
        dev, test = run_one(pd, args.epochs, args.batch_size, args.lr)
        summary[pd] = {'dev': dev, 'test': test}

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  p={k:<5}  dev={v["dev"]:.4f}  test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
