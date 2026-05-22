"""
hyperparameter_search.py — quick grid search over (lr, hidden) for a small MLP.

Use this as a starting point. For the full set of experiments, see the
`experiments/` folder.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mynn as nn
from experiments.common import load_mnist, eval_on_test


def main():
    np.random.seed(0)
    train, dev, test = load_mnist()

    grid = []
    for hidden in (256, 512):
        for lr in (5e-4, 1e-3, 2e-3):
            tag = f'h{hidden}_lr{lr}'
            print(f'\n=== {tag} ===')
            np.random.seed(0)
            model = nn.models.Model_MLP([784, hidden, 10], 'ReLU', init='he')
            opt = nn.optimizer.Adam(lr, model)
            loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
            runner = nn.runner.RunnerM(model, opt, nn.metric.accuracy, loss_fn,
                                       batch_size=128, verbose=False)
            runner.train(train, dev, num_epochs=3,
                         save_dir='./best_models/hp', save_name=f'{tag}.pickle')
            test_acc, *_ = eval_on_test(model, test)
            grid.append((tag, runner.best_score, test_acc))
            print(f'   dev={runner.best_score:.4f}  test={test_acc:.4f}')

    print('\nFinal grid:')
    for tag, dev_acc, test_acc in grid:
        print(f'  {tag:>20}  dev={dev_acc:.4f}  test={test_acc:.4f}')


if __name__ == '__main__':
    main()
