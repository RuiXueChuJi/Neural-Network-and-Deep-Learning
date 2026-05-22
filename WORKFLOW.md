# PJ1 Workflow Tracker

> Course: 神经网络与深度学习
> Project: PJ1 — MNIST classification with NumPy-only neural network framework
> Student: 22300680293
> Last updated: 2026/05/15

This document tracks progress through the project: **what step we are on**, **which files have been created/modified**, and **what experiments have been run**.

---

## Progress Bar

| Step | Phase | Status |
| :--: | :---- | :----: |
|  1  | Explore existing code & dataset                            |  ✅ Done  |
|  2  | Implement `mynn/op.py` (Linear, Loss, conv2D, …)            |  ✅ Done  |
|  3  | Implement `mynn/optimizer.py` (SGD, MomentGD, Adam, AdamW) |  ✅ Done  |
|  4  | Implement `mynn/lr_scheduler.py` (Step / MultiStep / Exp / Cosine) |  ✅ Done  |
|  5  | Implement `mynn/models.py` (MLP + CNN)                     |  ✅ Done  |
|  6  | Update `mynn/runner.py` (efficient eval, augmentation hook) |  ✅ Done  |
|  7  | Update `test_train.py` & `test_model.py`                   |  ✅ Done  |
|  8  | Run baseline MLP / final MLP                               |  ✅ Done  |
|  9  | Direction 1 — Optimization (3 experiments)                 |  ✅ Done  |
| 10  | Direction 2 — Regularization (3 experiments)               |  ✅ Done  |
| 11  | Run final CNN                                              |  ✅ Done  |
| 12  | Generate visualizations (24 figures)                       |  ✅ Done  |
| 13  | Write report `report/REPORT.md`                            |  ✅ Done  |
| 14  | Upload to GitHub & ModelScope                              |  ⏳ Pending — waiting for go-ahead |

---

## File Structure

```
HW2_22300680293/
├── WORKFLOW.md                 ← this tracker
├── PJ1/
│   ├── project_1.pdf
│   └── codes/
│       ├── README.md
│       ├── mynn/
│       │   ├── __init__.py
│       │   ├── op.py             ★ implemented (Linear, conv2D, MaxPool, Flatten, Dropout, BN, 5 activations, MultiCrossEntropyLoss)
│       │   ├── optimizer.py      ★ implemented (SGD, MomentGD/Nesterov, Adam, AdamW)
│       │   ├── lr_scheduler.py   ★ implemented (Step / MultiStep / Exp / Cosine / LinearWarmup)
│       │   ├── models.py         ★ implemented (Model_MLP, Model_CNN, save/load)
│       │   ├── runner.py         ★ improved (epoch eval, augmentation hook)
│       │   └── metric.py
│       ├── draw_tools/
│       │   ├── plot.py
│       │   └── draw.py
│       ├── dataset/MNIST/...
│       ├── test_train.py         ★ updated
│       ├── test_model.py         ★ updated (MLP + CNN, batched eval)
│       ├── hyperparameter_search.py    ★ implemented (lr × hidden grid)
│       ├── weight_visualization.py     ★ updated (MLP + CNN)
│       ├── experiments/                ★ NEW
│       │   ├── __init__.py
│       │   ├── common.py               # shared helpers + augmentations
│       │   ├── exp_d1_optimizer.py     # D1.1
│       │   ├── exp_d1_scheduler.py     # D1.2
│       │   ├── exp_d1_init.py          # D1.3
│       │   ├── exp_d2_weight_decay.py  # D2.1
│       │   ├── exp_d2_dropout.py       # D2.2
│       │   ├── exp_d2_augment.py       # D2.3
│       │   ├── train_final.py          # baseline / final_mlp / final_cnn
│       │   ├── run_all.py              # run every ablation sequentially
│       │   ├── make_figs.py            # build all figures used in report
│       │   ├── summarize.py            # CSV + Markdown summary
│       │   └── patch_report.py         # plug values into REPORT.md
│       ├── results/             ★ JSON logs from each run
│       ├── figs/                ★ in-codes figures (cleared)
│       ├── saved_models/        ★ baseline_mlp.pickle, final_mlp.pickle, final_cnn.pickle
│       └── best_models/         ★ best models per ablation
└── report/
    ├── REPORT.md                ★ final report (Markdown, 24 figures embedded)
    └── images/                  ★ 24 PNG figures
```

---

## Final Test Numbers

> Higher is better. Last year's reference report graded 91 capped at **97.89%**. We are at **98.81%**.

| Tag | Test acc |
| :-- | -------: |
| baseline_mlp (`[784,600,10]`, SGD lr=0.06, MultiStepLR) | **93.86%** |
| **final_mlp** (`[784,512,256,10]`, AdamW + warmup-cosine + dropout 0.2 + WD 1e-4 + light aug) | **98.81%** |
| **final_cnn** (Conv8→Conv16→FC128→10, AdamW + cosine + WD 1e-4 + dropout 0.2) | **98.79%** |

### Direction 1 — Optimization

| D1.1 Optimizer | Best Test |
| :--- | --: |
| SGD             | 94.71% |
| Momentum        | **97.85%** |
| Nesterov        | 97.62% |
| Adam            | 97.75% |
| AdamW           | 97.75% |

| D1.2 Scheduler | Best Test |
| :--- | --: |
| constant        | 97.75% |
| Step            | 97.59% |
| MultiStep       | 97.71% |
| Exponential     | 96.66% |
| **Cosine**      | **97.82%** |
| Warmup + Cosine | 97.69% |

| D1.3 Init | Best Test |
| :--- | --: |
| Normal(0, 0.01) | 98.00% |
| **Xavier**      | **98.09%** |
| He / Kaiming    | 97.83% |

### Direction 2 — Regularization

| D2.1 Weight decay λ | Best Test |
| :--- | --: |
| 0       | 97.75% |
| **1e-5** | **97.78%** |
| 1e-4    | 97.58% |
| 1e-3    | 97.13% |
| 1e-2    | 93.54% (over-regularized) |

| D2.2 Dropout p | Best Test |
| :--- | --: |
| 0.0     | 97.83% |
| 0.1     | 97.63% |
| **0.2** | **98.10%** |
| 0.3     | 97.90% |
| 0.5     | 97.68% |

| D2.3 Augmentation | Best Test |
| :--- | --: |
| none   | 97.75% |
| rotate ±10° | 98.01% |
| shift ±2 px | 98.03% |
| Gaussian noise | 97.82% |
| **combined**   | **98.13%** |

---

## How to Reproduce

```powershell
# from PJ1/codes/

# baseline + final models
python experiments/train_final.py --tag baseline_mlp
python experiments/train_final.py --tag final_mlp
python experiments/train_final.py --tag final_cnn

# all 6 ablations + final CNN at once
python experiments/run_all.py            # ~50–60 minutes total

# build figures + summary + patch report
python experiments/make_figs.py
python experiments/summarize.py
python experiments/patch_report.py

# evaluate any saved model on the test set
python test_model.py --model saved_models/final_mlp.pickle --kind mlp
python test_model.py --model saved_models/final_cnn.pickle --kind cnn

# weight visualizations
python weight_visualization.py --model saved_models/final_mlp.pickle --kind mlp
python weight_visualization.py --model saved_models/final_cnn.pickle --kind cnn
```

---

## Pending — to do when user signals

- [ ] Push the `codes/` folder to GitHub.
- [ ] Push the `saved_models/final_mlp.pickle` and `saved_models/final_cnn.pickle` files to ModelScope.
