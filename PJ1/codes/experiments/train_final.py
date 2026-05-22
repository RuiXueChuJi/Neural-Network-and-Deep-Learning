"""
Train the final / baseline / CNN models.

Run via:
    python experiments/train_final.py --tag baseline_mlp
    python experiments/train_final.py --tag final_mlp
    python experiments/train_final.py --tag final_cnn
"""

import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import (load_mnist, dump_history, save_results,
                                eval_on_test, make_augmentor)


def build(tag):
    """Return (model, optimizer, scheduler_factory(total_iters), loss_fn, augmentor, batch_size, epochs)."""
    if tag == 'baseline_mlp':
        m = nn.models.Model_MLP([784, 600, 10], 'ReLU', init='he')
        opt = nn.optimizer.SGD(0.06, m)
        def sched_fac(n):
            return nn.lr_scheduler.MultiStepLR(opt, milestones=[int(n*0.4), int(n*0.7), int(n*0.9)], gamma=0.3)
        loss = nn.op.MultiCrossEntropyLoss(model=m, max_classes=10)
        return m, opt, sched_fac, loss, None, 128, 8

    if tag == 'final_mlp':
        # Combined best: wider+deeper MLP, AdamW, cosine, dropout, weight-decay, light aug
        m = nn.models.Model_MLP([784, 512, 256, 10], 'ReLU',
                                lambda_list=[1e-4, 1e-4, 1e-4],
                                dropout=0.2, init='he')
        opt = nn.optimizer.AdamW(1e-3, m)
        def sched_fac(n):
            inner = nn.lr_scheduler.CosineAnnealingLR(opt, T_max=n - 200, eta_min=1e-5)
            return nn.lr_scheduler.LinearWarmup(opt, inner, warmup_iters=200, start_factor=0.1)
        loss = nn.op.MultiCrossEntropyLoss(model=m, max_classes=10)
        aug = make_augmentor(rotate_max=8, shift_max=2,
                             prob_rotate=0.4, prob_shift=0.4, seed=10)
        return m, opt, sched_fac, loss, aug, 128, 12

    if tag == 'final_cnn':
        m = nn.models.Model_CNN(in_channels=1, conv_channels=[8, 16],
                                kernel_size=3, fc_sizes=[128],
                                input_hw=(28, 28),
                                weight_decay_lambda=1e-4,
                                dropout=0.2, init='he')
        opt = nn.optimizer.AdamW(1e-3, m)
        def sched_fac(n):
            return nn.lr_scheduler.CosineAnnealingLR(opt, T_max=n, eta_min=1e-5)
        loss = nn.op.MultiCrossEntropyLoss(model=m, max_classes=10)
        return m, opt, sched_fac, loss, None, 64, 5

    raise ValueError(tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True,
                   choices=['baseline_mlp', 'final_mlp', 'final_cnn'])
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--seed', type=int, default=309)
    args = p.parse_args()

    np.random.seed(args.seed)
    train, dev, test = load_mnist(seed=args.seed)

    model, optimizer, sched_factory, loss_fn, aug, batch_size, default_epochs = build(args.tag)
    epochs = args.epochs or default_epochs
    n_iters = ((train[0].shape[0] + batch_size - 1) // batch_size) * epochs
    scheduler = sched_factory(n_iters)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                               batch_size=batch_size, scheduler=scheduler, augmentor=aug)
    runner.train(train, dev, num_epochs=epochs,
                 save_dir=save_dir, save_name=f'{args.tag}.pickle')

    # Reload best and evaluate on test
    if args.tag == 'final_cnn':
        loaded = nn.models.Model_CNN()
    else:
        loaded = nn.models.Model_MLP()
    loaded.load_model(os.path.join(save_dir, f'{args.tag}.pickle'))
    test_acc, _, _ = eval_on_test(loaded, test)

    history = dump_history(runner)
    history['tag'] = args.tag
    history['epochs'] = epochs
    history['batch_size'] = batch_size
    history['test_acc'] = test_acc
    save_results(args.tag, history)

    print(f'\n=== {args.tag} ===')
    print(f'  best dev acc = {runner.best_score:.4f}')
    print(f'  test acc     = {test_acc:.4f}')


if __name__ == '__main__':
    main()
