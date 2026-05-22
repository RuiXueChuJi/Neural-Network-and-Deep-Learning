"""
make_figs.py — read everything in ../results/*.json and produce the figures
referenced from the report.

Outputs go to ../../report/images/ (relative to this script).
"""

import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mynn as nn
from experiments.common import load_mnist, eval_on_test


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   13,
    'legend.fontsize':  10,
    'savefig.bbox':     'tight',
    'savefig.dpi':      150,
})

PALETTE = ['#3F88C5', '#E94F37', '#44BBA4', '#F6AE2D', '#A663CC',
           '#2E2C2F', '#7A9E9F', '#D7263D', '#1B998B', '#F46036']

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
def plot_train_dev(name, savepath, title=None):
    h = load(name)
    if h is None:
        print(f'  [skip] {name}')
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(h['epoch_train_loss']) + 1)
    axes[0].plot(epochs, h['epoch_train_loss'], color=PALETTE[0], marker='o', label='train')
    axes[0].plot(epochs, h['epoch_dev_loss'],   color=PALETTE[1], marker='s', label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
    axes[0].set_title('Loss')
    axes[0].legend(loc='upper right')

    axes[1].plot(epochs, h['epoch_train_acc'], color=PALETTE[0], marker='o', label='train')
    axes[1].plot(epochs, h['epoch_dev_acc'],   color=PALETTE[1], marker='s', label='val')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(loc='lower right')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, savepath))
    plt.close(fig)
    print(f'  wrote {savepath}')


def plot_overlay(names, labels, savepath, title, ylabel='val acc', metric='epoch_dev_acc'):
    """Overlay one curve per experiment for comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (n, lbl) in enumerate(zip(names, labels)):
        h = load(n)
        if h is None or len(h[metric]) == 0:
            continue
        epochs = np.arange(1, len(h[metric]) + 1)
        ax.plot(epochs, h[metric], color=PALETTE[i % len(PALETTE)],
                marker='o', linewidth=2, label=lbl)
    ax.set_xlabel('epoch'); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, savepath))
    plt.close(fig)
    print(f'  wrote {savepath}')


def plot_bar(names, labels, savepath, title, key='test_acc', ylabel='test accuracy'):
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = []
    keep_labels = []
    for n, lbl in zip(names, labels):
        h = load(n)
        if h is None or key not in h:
            continue
        vals.append(h[key])
        keep_labels.append(lbl)
    bars = ax.bar(keep_labels, vals,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(vals))],
                  edgecolor='#222', linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f'{v*100:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([min(vals) * 0.97, max(vals) * 1.01 + 0.005])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, savepath))
    plt.close(fig)
    print(f'  wrote {savepath}')


# ----------------------------------------------------------------------------
def make_optimizer_figs():
    names = ['d1_optimizer_sgd', 'd1_optimizer_momentum', 'd1_optimizer_nesterov',
             'd1_optimizer_adam', 'd1_optimizer_adamw']
    labels = ['SGD', 'Momentum', 'Nesterov', 'Adam', 'AdamW']
    plot_overlay(names, labels, 'd1_optimizer_dev.png',
                 'Direction 1.1 — Optimizer comparison (val accuracy)',
                 ylabel='val accuracy')
    plot_overlay(names, labels, 'd1_optimizer_loss.png',
                 'Direction 1.1 — Optimizer comparison (training loss)',
                 ylabel='train loss', metric='epoch_train_loss')
    plot_bar(names, labels, 'd1_optimizer_test.png',
             'Direction 1.1 — Test accuracy by optimizer')


def make_scheduler_figs():
    names = ['d1_scheduler_constant', 'd1_scheduler_step', 'd1_scheduler_multistep',
             'd1_scheduler_exp', 'd1_scheduler_cosine', 'd1_scheduler_warmup_cosine']
    labels = ['constant', 'step', 'multistep', 'exp', 'cosine', 'warmup+cosine']
    plot_overlay(names, labels, 'd1_scheduler_dev.png',
                 'Direction 1.2 — LR scheduler comparison (val accuracy)')
    plot_bar(names, labels, 'd1_scheduler_test.png',
             'Direction 1.2 — Test accuracy by scheduler')

    # Plot the LR profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (n, lbl) in enumerate(zip(names, labels)):
        h = load(n)
        if h is None or len(h.get('lr_history', [])) == 0:
            continue
        ax.plot(h['lr_history'], color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=lbl)
    ax.set_xlabel('iteration'); ax.set_ylabel('learning rate')
    ax.set_yscale('log')
    ax.set_title('LR profile of each scheduler')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'd1_scheduler_lr.png'))
    plt.close(fig)
    print('  wrote d1_scheduler_lr.png')


def make_init_figs():
    names = ['d1_init_normal', 'd1_init_xavier', 'd1_init_he']
    labels = ['Normal(0, 0.01)', 'Xavier', 'He']
    plot_overlay(names, labels, 'd1_init_dev.png',
                 'Direction 1.3 — Initialization comparison (val accuracy)')
    plot_overlay(names, labels, 'd1_init_loss.png',
                 'Direction 1.3 — Initialization comparison (train loss)',
                 ylabel='train loss', metric='epoch_train_loss')
    plot_bar(names, labels, 'd1_init_test.png',
             'Direction 1.3 — Test accuracy by init scheme')


def make_wd_figs():
    lams = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    names = [f'd2_wd_{lam}' for lam in lams]
    labels = [f'λ={lam:g}' for lam in lams]
    plot_overlay(names, labels, 'd2_wd_dev.png',
                 'Direction 2.1 — Weight-decay sweep (val accuracy)')
    # gap = train acc - val acc to show overfitting
    fig, ax = plt.subplots(figsize=(8, 5))
    test_accs = []
    keep = []
    for lam, lbl in zip(lams, labels):
        h = load(f'd2_wd_{lam}')
        if h is None: continue
        gap = np.array(h['epoch_train_acc']) - np.array(h['epoch_dev_acc'])
        ax.plot(np.arange(1, len(gap) + 1), gap, marker='o', label=lbl)
        test_accs.append(h.get('test_acc', 0))
        keep.append(lbl)
    ax.set_xlabel('epoch'); ax.set_ylabel('train_acc − val_acc (overfit gap)')
    ax.set_title('Direction 2.1 — Overfitting gap shrinks with weight decay')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'd2_wd_gap.png'))
    plt.close(fig)
    print('  wrote d2_wd_gap.png')

    plot_bar(names, labels, 'd2_wd_test.png',
             'Direction 2.1 — Test accuracy across weight-decay strengths')


def make_dropout_figs():
    ps = [0.0, 0.1, 0.2, 0.3, 0.5]
    names = [f'd2_dropout_{p}' for p in ps]
    labels = [f'p={p}' for p in ps]
    plot_overlay(names, labels, 'd2_dropout_dev.png',
                 'Direction 2.2 — Dropout sweep (val accuracy)')
    plot_bar(names, labels, 'd2_dropout_test.png',
             'Direction 2.2 — Test accuracy across dropout rates')


def make_aug_figs():
    aug_names = ['none', 'rotate', 'shift', 'noise', 'combined']
    names = [f'd2_aug_{a}' for a in aug_names]
    labels = aug_names
    plot_overlay(names, labels, 'd2_aug_dev.png',
                 'Direction 2.3 — Augmentation comparison (val accuracy)')
    plot_bar(names, labels, 'd2_aug_test.png',
             'Direction 2.3 — Test accuracy by augmentation')


def make_baseline_final_figs():
    plot_train_dev('baseline_mlp', 'baseline_mlp.png',
                   'Baseline MLP — [784, 600, 10] + SGD + MultiStepLR')
    plot_train_dev('final_mlp', 'final_mlp.png',
                   'Final MLP — AdamW + warmup-cosine + dropout + WD + light aug')
    plot_train_dev('final_cnn', 'final_cnn.png',
                   'Final CNN — Conv8→Conv16 + AdamW + cosine')


# ----------------------------------------------------------------------------
def visualize_mlp_weights(model_path, save_name='final_mlp_weights.png',
                          first_layer_size=(28, 28), n_show=64):
    """Show the first-layer weight columns reshaped to 28×28 images."""
    if not os.path.exists(model_path):
        print(f'  [skip weights] no {model_path}')
        return
    m = nn.models.Model_MLP()
    m.load_model(model_path)
    # find first Linear layer
    W0 = None
    for layer in m.layers:
        if hasattr(layer, 'params') and 'W' in layer.params and layer.__class__.__name__ == 'Linear':
            W0 = layer.params['W']
            break
    if W0 is None or W0.shape[0] != first_layer_size[0] * first_layer_size[1]:
        print('  [skip weights] first layer not 784->H')
        return
    n_show = min(n_show, W0.shape[1])
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle('Final MLP — first-layer weight images (each panel = one hidden unit)',
                 fontsize=14, fontweight='bold')
    cmap = plt.cm.RdBu_r
    for k in range(n_show):
        r, c = k // 8, k % 8
        w = W0[:, k].reshape(*first_layer_size)
        vmax = np.max(np.abs(w))
        axes[r, c].imshow(w, cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for k in range(n_show, 64):
        r, c = k // 8, k % 8
        axes[r, c].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, save_name))
    plt.close(fig)
    print(f'  wrote {save_name}')


def visualize_cnn_kernels(model_path, save_name='final_cnn_kernels.png'):
    if not os.path.exists(model_path):
        print(f'  [skip kernels] no {model_path}')
        return
    m = nn.models.Model_CNN()
    m.load_model(model_path)
    # Get first conv W: shape [out, in, k, k]
    convs = [l for l in m.layers if l.__class__.__name__ == 'conv2D']
    if not convs:
        print('  [skip kernels] no conv layers'); return

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle('Final CNN — first conv-layer kernels', fontsize=14, fontweight='bold')
    W = convs[0].params['W']  # [8, 1, 3, 3]
    n = min(8, W.shape[0])
    for k in range(n):
        kern = W[k, 0]
        vmax = np.max(np.abs(kern))
        axes[0, k].imshow(kern, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[0, k].set_title(f'kern {k}', fontsize=9)

    if len(convs) >= 2:
        W2 = convs[1].params['W']  # [16, 8, 3, 3]
        # Show 8 kernels averaged across input channels
        for k in range(min(8, W2.shape[0])):
            kern = W2[k].mean(axis=0)
            vmax = np.max(np.abs(kern))
            axes[1, k].imshow(kern, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
            axes[1, k].set_title(f'L2 kern {k}', fontsize=9)
    else:
        for k in range(8):
            axes[1, k].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, save_name))
    plt.close(fig)
    print(f'  wrote {save_name}')


def confusion_matrix(model_path, kind, save_name):
    if not os.path.exists(model_path):
        print(f'  [skip CM] no {model_path}')
        return
    if kind == 'mlp':
        m = nn.models.Model_MLP()
    else:
        m = nn.models.Model_CNN()
    m.load_model(model_path)
    if hasattr(m, 'eval'):
        m.eval()
    _, _, test = load_mnist()
    test_acc, logits, pred = eval_on_test(m, test)
    y = test[1]
    K = 10
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    for i in range(K):
        for j in range(K):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v),
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else '#222',
                    fontsize=8)
    ax.set_title(f'Confusion matrix — test acc {test_acc*100:.2f}%')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, save_name))
    plt.close(fig)
    print(f'  wrote {save_name}')


def make_summary_dashboard():
    """Big single image showing the highlights for the report cover."""
    expected = ['baseline_mlp', 'final_mlp', 'final_cnn']
    have = [n for n in expected if load(n) is not None]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5 * len(have), 4.5))
    if len(have) == 1:
        axes = [axes]
    for ax, n in zip(axes, have):
        h = load(n)
        epochs = np.arange(1, len(h['epoch_train_acc']) + 1)
        ax.plot(epochs, h['epoch_train_acc'], color=PALETTE[0], linewidth=2, marker='o', label='train')
        ax.plot(epochs, h['epoch_dev_acc'],   color=PALETTE[1], linewidth=2, marker='s', label='val')
        ax.set_title(f'{n}\nval={h["best_dev_acc"]*100:.2f}%, test={h.get("test_acc",0)*100:.2f}%',
                     fontsize=12)
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.legend(loc='lower right')
        ax.set_ylim([0.85, 1.0])
    fig.suptitle('PJ1 — Final model summary', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'summary_dashboard.png'))
    plt.close(fig)
    print('  wrote summary_dashboard.png')


def main():
    print('--- Direction 1 ---')
    make_optimizer_figs()
    make_scheduler_figs()
    make_init_figs()

    print('--- Direction 2 ---')
    make_wd_figs()
    make_dropout_figs()
    make_aug_figs()

    print('--- Final / baseline ---')
    make_baseline_final_figs()
    make_summary_dashboard()

    print('--- Weight visualizations ---')
    visualize_mlp_weights(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_mlp.pickle'),
        'final_mlp_weights.png')
    visualize_cnn_kernels(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_cnn.pickle'),
        'final_cnn_kernels.png')

    print('--- Confusion matrices ---')
    confusion_matrix(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_mlp.pickle'),
        'mlp', 'final_mlp_cm.png')
    confusion_matrix(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'final_cnn.pickle'),
        'cnn', 'final_cnn_cm.png')


if __name__ == '__main__':
    main()
