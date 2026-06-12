"""Loss-landscape experiments for VGG-A with and without BatchNorm."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from utils.device import get_device, set_device_seed

DEFAULT_LRS = [1e-3, 2e-3, 1e-4, 5e-4]


def set_random_seeds(seed_value=0, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    set_device_seed(seed_value, device)
    if device != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    return correct / total


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / total


def train(model, optimizer, criterion, train_loader, val_loader, device, scheduler=None,
          epochs_n=20, best_model_path=None):
    model.to(device)
    losses_list = []
    grads = []
    epoch_rows = []
    max_val_accuracy = 0.0

    for epoch in range(1, epochs_n + 1):
        model.train()
        epoch_loss = 0.0
        epoch_total = 0
        start = time.perf_counter()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            loss.backward()

            grad_tensor = None
            for parameter in reversed(list(model.parameters())):
                if parameter.grad is not None:
                    grad_tensor = parameter.grad.detach().norm().item()
                    break
            grads.append(grad_tensor)
            optimizer.step()

            batch_size = targets.size(0)
            losses_list.append(loss.item())
            epoch_loss += loss.item() * batch_size
            epoch_total += batch_size

        if scheduler is not None:
            scheduler.step()

        train_loss = epoch_loss / epoch_total
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, val_loader, device)
        seconds = time.perf_counter() - start
        if best_model_path is not None and val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
            "epoch_seconds": seconds,
        }
        epoch_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    return np.asarray(losses_list, dtype=np.float32), np.asarray(grads, dtype=np.float32), epoch_rows


def align_loss_curves(curves):
    min_len = min(len(curve) for curve in curves)
    if min_len == 0:
        raise ValueError("At least one loss curve is empty")
    return np.stack([curve[:min_len] for curve in curves], axis=0)


def compute_envelope(curves):
    aligned = align_loss_curves(curves)
    min_curve = aligned.min(axis=0)
    max_curve = aligned.max(axis=0)
    mean_curve = aligned.mean(axis=0)
    return min_curve, max_curve, mean_curve


def write_epoch_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_lr_sweep(model_ctor, model_name, lrs, train_loader, val_loader, epochs, device, output_dir, seed):
    criterion = nn.CrossEntropyLoss()
    curves = []
    run_summaries = []
    for lr in lrs:
        set_random_seeds(seed, device.type)
        model = model_ctor()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_dir = output_dir / model_name / f"lr_{lr:g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        loss_curve, grad_curve, epoch_rows = train(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs_n=epochs, best_model_path=run_dir / "best_model.pt")
        np.save(run_dir / "step_losses.npy", loss_curve)
        np.save(run_dir / "grad_norms.npy", grad_curve)
        write_epoch_rows(run_dir / "epoch_metrics.csv", epoch_rows)
        curves.append(loss_curve)
        run_summaries.append({
            "model": model_name,
            "lr": lr,
            "steps": int(len(loss_curve)),
            "final_loss": float(loss_curve[-1]),
            "best_val_acc": max(row["val_acc"] for row in epoch_rows),
            "parameters": int(get_number_of_parameters(model)),
        })
    min_curve, max_curve, mean_curve = compute_envelope(curves)
    envelope_dir = output_dir / model_name
    np.save(envelope_dir / "min_curve.npy", min_curve)
    np.save(envelope_dir / "max_curve.npy", max_curve)
    np.save(envelope_dir / "mean_curve.npy", mean_curve)
    return {
        "model": model_name,
        "curves": curves,
        "min_curve": min_curve,
        "max_curve": max_curve,
        "mean_curve": mean_curve,
        "runs": run_summaries,
    }


def plot_loss_envelope(min_curve, max_curve, mean_curve, label, color, output_path):
    steps = np.arange(len(min_curve))
    plt.figure(figsize=(8, 5))
    plt.fill_between(steps, min_curve, max_curve, alpha=0.25, color=color, label=f"{label} min-max")
    plt.plot(steps, mean_curve, color=color, linewidth=1.5, label=f"{label} mean")
    plt.xlabel("Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Loss Landscape Envelope: {label}")
    plt.legend()
    plt.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bn_vs_plain(plain, bn, output_path):
    common_len = min(len(plain["min_curve"]), len(bn["min_curve"]))
    steps = np.arange(common_len)
    plt.figure(figsize=(9, 5.5))
    plt.fill_between(steps, plain["min_curve"][:common_len], plain["max_curve"][:common_len],
                     alpha=0.25, color="tab:green", label="Standard VGG min-max")
    plt.plot(steps, plain["mean_curve"][:common_len], color="tab:green", linewidth=1.2,
             label="Standard VGG mean")
    plt.fill_between(steps, bn["min_curve"][:common_len], bn["max_curve"][:common_len],
                     alpha=0.25, color="tab:red", label="VGG + BatchNorm min-max")
    plt.plot(steps, bn["mean_curve"][:common_len], color="tab:red", linewidth=1.2,
             label="VGG + BatchNorm mean")
    plt.xlabel("Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Landscape: VGG-A vs VGG-A + BatchNorm")
    plt.legend()
    plt.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="VGG-A BN loss-landscape sweep")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="reports/task2_loss_landscape")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"[device] using {device}", flush=True)
    set_random_seeds(args.seed, device.type)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_cifar_loader(root=args.data_root, batch_size=args.batch_size, train=True,
                                    shuffle=True, num_workers=args.num_workers, n_items=args.n_items)
    val_loader = get_cifar_loader(root=args.data_root, batch_size=args.batch_size, train=False,
                                  shuffle=False, num_workers=args.num_workers, n_items=args.n_items)

    config = vars(args).copy()
    config["device"] = str(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    plain = run_lr_sweep(VGG_A, "vgg_a", args.lrs, train_loader, val_loader,
                         args.epochs, device, output_dir, args.seed)
    bn = run_lr_sweep(VGG_A_BatchNorm, "vgg_a_batchnorm", args.lrs, train_loader, val_loader,
                      args.epochs, device, output_dir, args.seed)

    plot_loss_envelope(plain["min_curve"], plain["max_curve"], plain["mean_curve"],
                       "Standard VGG-A", "tab:green", output_dir / "vgg_a_envelope.png")
    plot_loss_envelope(bn["min_curve"], bn["max_curve"], bn["mean_curve"],
                       "VGG-A + BatchNorm", "tab:red", output_dir / "vgg_a_batchnorm_envelope.png")
    plot_bn_vs_plain(plain, bn, output_dir / "loss_landscape_bn_vs_plain.png")

    summary = {
        "learning_rates": args.lrs,
        "plain_runs": plain["runs"],
        "batchnorm_runs": bn["runs"],
        "plain_band_mean_width": float(np.mean(plain["max_curve"] - plain["min_curve"])),
        "batchnorm_band_mean_width": float(np.mean(bn["max_curve"] - bn["min_curve"])),
        "combined_figure": str(output_dir / "loss_landscape_bn_vs_plain.png"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
