# Neural Network and Deep Learning — PJ1 & PJ2
PJ1文件夹之外的所有data models reports等内容都是PJ2的

> Course: 神经网络与深度学习 (Neural Network and Deep Learning)
> Project 1: MNIST digit recognition with a NumPy-only neural network framework
> Author: 严瑞琪 (22300680293)

This repository contains the code and report for **PJ1**: building an MLP / CNN
framework from scratch in NumPy and training it on MNIST.

- Final MLP `[784, 512, 256, 10]` — **test 98.81%** (AdamW + warmup-cosine + dropout 0.2 + WD 1e-4 + light aug)
- Final CNN `Conv8 → Conv16 → FC128 → 10` — **test 98.79%** (AdamW + cosine + dropout 0.2 + WD 1e-4)
- Baseline MLP `[784, 600, 10]` — test 93.86% (SGD + MultiStepLR)

## Trained model weights

The trained model checkpoints (`baseline_mlp.pickle`, `final_mlp.pickle`,
`final_cnn.pickle`) and all per-experiment best models are hosted on
ModelScope:

**https://www.modelscope.cn/models/RuiXueChuJi/NN-DL-HW2/summary**

Per the assignment instructions, the dataset and model weights are **not**
included in this repository.

## Repository layout

```
HW2_22300680293/
├── WORKFLOW.md                       # progress tracker
├── PJ1/
│   ├── project_1.pdf                 # assignment handout
│   └── codes/
│       ├── README.md                 # original course README
│       ├── mynn/                     # the NumPy framework
│       │   ├── op.py                 # Linear, conv2D, MaxPool, Flatten, Dropout, BN, activations, loss
│       │   ├── optimizer.py          # SGD / MomentGD / Adam / AdamW
│       │   ├── lr_scheduler.py       # Step / MultiStep / Exp / Cosine / LinearWarmup
│       │   ├── models.py             # Model_MLP, Model_CNN
│       │   ├── runner.py             # training loop with epoch eval + augmentation hook
│       │   └── metric.py
│       ├── experiments/              # all ablations + final-model training
│       ├── draw_tools/               # plotting helpers
│       ├── test_train.py             # quick training entry point
│       ├── test_model.py             # evaluate any saved checkpoint
│       ├── weight_visualization.py   # MLP / CNN visualizers
│       ├── hyperparameter_search.py
│       └── dataset_explore.ipynb
└── report/
    ├── REPORT.md                     # the project report (read this!)
    ├── REPORT.html
    └── images/                       # all figures referenced by the report
```

## How to reproduce

Download the MNIST dataset into `PJ1/codes/dataset/MNIST/` (see
`dataset_explore.ipynb`), then download the model weights from the
ModelScope link above into `PJ1/codes/saved_models/`.

```powershell
# from PJ1/codes/

# train baseline + final MLP + final CNN
python experiments/train_final.py --tag baseline_mlp
python experiments/train_final.py --tag final_mlp
python experiments/train_final.py --tag final_cnn

# evaluate
python test_model.py --model saved_models/final_mlp.pickle --kind mlp
python test_model.py --model saved_models/final_cnn.pickle --kind cnn

# run every ablation (D1 + D2)
python experiments/run_all.py

# rebuild every figure
python experiments/make_figs.py
```

See [report/REPORT.md](report/REPORT.md) for the full write-up, results
tables, and visualizations.
