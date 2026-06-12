"""Plot gradient-predictiveness and beta-smoothness envelopes for VGG-A
with and without BatchNorm, following Santurkar et al. (2018).

We only logged the per-step gradient L2-norm (a scalar), so:
- gradient predictiveness  -> min/max envelope of grad-norm across the 4 LRs
- beta-smoothness (proxy)   -> envelope of |g_{t+1}-g_t| (step-to-step change)
Both use the same multi-LR fill_between construction as the loss landscape.
"""
import glob
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "reports/task2_loss_landscape"


def load_grads(model):
    out = []
    for d in sorted(glob.glob(f"{ROOT}/{model}/lr_*")):
        out.append(np.load(os.path.join(d, "grad_norms.npy")))
    return np.stack(out, axis=0)  # (n_lr, steps)


def envelopes(stack):
    return stack.min(axis=0), stack.max(axis=0), stack.mean(axis=0)


plain = load_grads("vgg_a")
bn = load_grads("vgg_a_batchnorm")
steps = np.arange(plain.shape[1])

# ---- Figure 1: gradient predictiveness (grad-norm envelope) ----
p_min, p_max, p_mean = envelopes(plain)
b_min, b_max, b_mean = envelopes(bn)

plt.figure(figsize=(8, 5))
plt.fill_between(steps, p_min, p_max, color="tab:green", alpha=0.30, label="Standard VGG (no BN)")
plt.fill_between(steps, b_min, b_max, color="tab:red", alpha=0.30, label="Standard VGG + BN")
plt.plot(steps, p_mean, color="tab:green", lw=0.8)
plt.plot(steps, b_mean, color="tab:red", lw=0.8)
plt.xlabel("Training Step")
plt.ylabel("Gradient L2-norm")
plt.title("Gradient Predictiveness: BN vs No-BN (envelope over LRs)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{ROOT}/grad_predictiveness_bn_vs_plain.png", dpi=220)
plt.close()

# ---- Figure 2: beta-smoothness proxy (|delta grad| envelope) ----
plain_d = np.abs(np.diff(plain, axis=1))
bn_d = np.abs(np.diff(bn, axis=1))
ds = np.arange(plain_d.shape[1])
pd_min, pd_max, pd_mean = envelopes(plain_d)
bd_min, bd_max, bd_mean = envelopes(bn_d)

plt.figure(figsize=(8, 5))
plt.fill_between(ds, pd_min, pd_max, color="tab:green", alpha=0.30, label="Standard VGG (no BN)")
plt.fill_between(ds, bd_min, bd_max, color="tab:red", alpha=0.30, label="Standard VGG + BN")
plt.xlabel("Training Step")
plt.ylabel(r"$|\,\nabla L_{t+1}-\nabla L_t\,|$  (grad-norm change)")
plt.title(r'"Effective $\beta$-smoothness" proxy: BN vs No-BN')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{ROOT}/beta_smoothness_bn_vs_plain.png", dpi=220)
plt.close()

# ---- stats for report ----
def band(stack):
    mn, mx, _ = envelopes(stack)
    return float((mx - mn).mean())

print("grad_pred_band_plain", round(band(plain), 4))
print("grad_pred_band_bn", round(band(bn), 4))
print("grad_norm_max_plain", round(float(plain.max()), 4))
print("grad_norm_max_bn", round(float(bn.max()), 4))
print("beta_band_plain", round(band(plain_d), 4))
print("beta_band_bn", round(band(bn_d), 4))
print("beta_max_plain", round(float(plain_d.max()), 4))
print("beta_max_bn", round(float(bn_d.max()), 4))
print("saved:", f"{ROOT}/grad_predictiveness_bn_vs_plain.png", f"{ROOT}/beta_smoothness_bn_vs_plain.png")
