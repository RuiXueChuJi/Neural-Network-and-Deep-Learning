"""
run_all.py — single entry point that runs every experiment sequentially.

Writes per-experiment JSON results into ../results/ and best models into
../best_models/<exp_tag>/.
"""

import os, sys, time, json, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all experiment runners
from experiments import exp_d1_optimizer
from experiments import exp_d1_scheduler
from experiments import exp_d1_init
from experiments import exp_d2_weight_decay
from experiments import exp_d2_dropout
from experiments import exp_d2_augment
from experiments import train_final


def section(title):
    line = '=' * 78
    print(f'\n{line}\n  {title}\n{line}', flush=True)


def safe_run(label, fn, *args, **kwargs):
    print(f'\n--- starting: {label} ---', flush=True)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
    except Exception:
        print(f'!! {label} crashed:')
        traceback.print_exc()
    print(f'--- finished {label} in {time.time()-t0:.1f}s ---', flush=True)


def main():
    EPOCHS = int(os.environ.get('EPOCHS', '6'))
    BATCH = int(os.environ.get('BATCH', '128'))

    # Replace argparse args by directly calling each module's `run_one`
    # Each experiment file has a `main()` that uses argparse — easier to call
    # the inner functions directly.

    # ---------- Baseline + Final ----------
    section('Baseline MLP (SGD + MultiStepLR)')
    sys.argv = ['x', '--tag', 'baseline_mlp', '--epochs', str(EPOCHS)]
    safe_run('train_final baseline_mlp', train_final.main)

    section('Final MLP (AdamW + cosine + warmup + dropout + WD + light aug)')
    sys.argv = ['x', '--tag', 'final_mlp', '--epochs', str(EPOCHS + 4)]
    safe_run('train_final final_mlp', train_final.main)

    # ---------- D1.1 Optimizer ----------
    section('D1.1 Optimizer Comparison')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d1_optimizer', exp_d1_optimizer.main)

    # ---------- D1.2 Scheduler ----------
    section('D1.2 Scheduler Comparison')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d1_scheduler', exp_d1_scheduler.main)

    # ---------- D1.3 Init ----------
    section('D1.3 Initialization Comparison')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d1_init', exp_d1_init.main)

    # ---------- D2.1 Weight Decay ----------
    section('D2.1 Weight Decay Sweep')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d2_weight_decay', exp_d2_weight_decay.main)

    # ---------- D2.2 Dropout ----------
    section('D2.2 Dropout Sweep')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d2_dropout', exp_d2_dropout.main)

    # ---------- D2.3 Augmentation ----------
    section('D2.3 Augmentation Comparison')
    sys.argv = ['x', '--epochs', str(EPOCHS), '--batch_size', str(BATCH)]
    safe_run('exp_d2_augment', exp_d2_augment.main)

    # ---------- CNN ----------
    section('Final CNN')
    sys.argv = ['x', '--tag', 'final_cnn', '--epochs', str(max(3, EPOCHS - 2))]
    safe_run('train_final final_cnn', train_final.main)

    print('\nALL DONE', flush=True)


if __name__ == '__main__':
    main()
