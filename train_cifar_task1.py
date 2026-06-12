"""Train configurable CIFAR-10 models for Project-2 Task 1."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data.loaders import get_cifar_loader
from models.cifar_net import ConfigurableCIFARNet, count_parameters, parse_filters
from utils.device import get_device, set_device_seed


def seed_everything(seed, device=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None:
        set_device_seed(seed, device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


class LabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothing CE that works on torch<1.10 (no built-in arg)."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


def build_loss(name, label_smoothing):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        # Use native arg when available (torch>=1.10), else portable fallback.
        try:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        except TypeError:
            return LabelSmoothingCrossEntropy(label_smoothing)
    if name == "ce_weight_decay":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {name}")


def build_optimizer(name, model, lr, weight_decay, momentum):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def accuracy_from_logits(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).sum().item(), targets.numel()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        batch_correct, batch_total = accuracy_from_logits(logits, targets)
        correct += batch_correct
        total += batch_total
    elapsed = time.perf_counter() - start
    return total_loss / total, correct / total, elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        batch_correct, batch_total = accuracy_from_logits(logits, targets)
        correct += batch_correct
        total += batch_total
    return total_loss / total, correct / total


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Configurable CIFAR-10 Project-2 trainer")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="reports/task1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--filters", default="64,128,256")
    parser.add_argument("--activation", choices=["relu", "gelu", "leaky_relu", "elu"], default="relu")
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-residual", action="store_true")
    parser.add_argument("--loss", choices=["ce", "label_smoothing", "ce_weight_decay"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"[device] using {device}", flush=True)
    seed_everything(args.seed, device)
    filters = parse_filters(args.filters)
    run_name = args.run_name or (
        f"f{'-'.join(map(str, filters))}_{args.activation}_{args.loss}_{args.optimizer}"
        f"_bn{int(args.use_bn)}_do{args.dropout}_res{int(args.use_residual)}"
    )
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_cifar_loader(root=args.data_root, batch_size=args.batch_size, train=True,
                                    shuffle=True, num_workers=args.num_workers, n_items=args.n_items)
    val_loader = get_cifar_loader(root=args.data_root, batch_size=args.batch_size, train=False,
                                  shuffle=False, num_workers=args.num_workers, n_items=args.n_items)

    model = ConfigurableCIFARNet(filters=filters, activation=args.activation,
                                 use_batchnorm=args.use_bn, dropout_p=args.dropout,
                                 use_residual=args.use_residual).to(device)
    criterion = build_loss(args.loss, args.label_smoothing)
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay, args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "filters": list(filters),
        "parameters": count_parameters(model),
    })
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    rows = []
    best_acc = -1.0
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, seconds = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_error": 1.0 - val_acc,
            "epoch_seconds": seconds,
            "lr": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "best_val_acc": best_acc,
                "epoch": epoch,
            }, best_path)

    write_metrics(output_dir / "metrics.csv", rows)
    summary = {
        "run_name": run_name,
        "best_val_acc": best_acc,
        "best_test_error": 1.0 - best_acc,
        "best_model": str(best_path),
        "metrics": str(output_dir / "metrics.csv"),
        "parameters": count_parameters(model),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in rows])),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
