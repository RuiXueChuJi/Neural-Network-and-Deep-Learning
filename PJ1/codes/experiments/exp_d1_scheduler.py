"""
Direction 1 — Experiment 1.2: Learning-rate scheduler comparison.

Trains the same MLP with the same Adam optimizer, varying only the schedule:
    - constant
    - StepLR
    - MultiStepLR
    - ExponentialLR
    - CosineAnnealingLR
    - LinearWarmup + Cosine
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import load_mnist, dump_history, save_results, eval_on_test


SIZE_LIST = [784, 512, 10]


def build_scheduler(name, optimizer, total_iters):
    if name == 'constant':
        return None
    if name == 'step':
        return nn.lr_scheduler.StepLR(optimizer, step_size=total_iters // 4, gamma=0.5)
    if name == 'multistep':
        ms = [int(total_iters * f) for f in (0.4, 0.7, 0.9)]
        return nn.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.3)
    if name == 'exp':
        # decay so lr at end is roughly 1e-2 of start
        gamma = 0.01 ** (1.0 / max(1, total_iters))
        return nn.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if name == 'cosine':
        return nn.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-5)
    if name == 'warmup_cosine':
        inner = nn.lr_scheduler.CosineAnnealingLR(optimizer,
                                                  T_max=total_iters - 200, eta_min=1e-5)
        return nn.lr_scheduler.LinearWarmup(optimizer, inner, warmup_iters=200, start_factor=0.1)
    raise ValueError(name)


def run_one(name, lr, epochs, batch_size, seed=309):
    np.random.seed(seed)
    train, dev, test = load_mnist(seed=seed)
    n_iters = ((train[0].shape[0] + batch_size - 1) // batch_size) * epochs

    model = nn.models.Model_MLP(SIZE_LIST, 'ReLU', init='he')
    optimizer = nn.optimizer.Adam(lr, model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    scheduler = build_scheduler(name, optimizer, n_iters)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size, scheduler=scheduler)
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'best_models', f'd1_sched_{name}')
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'best_{name}.pickle')
    model.load_model(os.path.join(save_dir, f'best_{name}.pickle'))
    test_acc, *_ = eval_on_test(model, test)

    h = dump_history(runner)
    h['name'] = name
    h['lr'] = lr
    h['test_acc'] = test_acc
    save_results(f'd1_scheduler_{name}', h)
    print(f'  -> [{name}] dev={runner.best_score:.4f} test={test_acc:.4f}')
    return runner.best_score, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    summary = {}
    for name in ['constant', 'step', 'multistep', 'exp', 'cosine', 'warmup_cosine']:
        print(f'\n=== Scheduler: {name} ===')
        dev, test = run_one(name, args.lr, args.epochs, args.batch_size)
        summary[name] = {'dev': dev, 'test': test}

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  {k:>14}  dev={v["dev"]:.4f}  test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
