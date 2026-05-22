"""
runner.py — training driver.

Improvements over the example runner:
  - Evaluation is run per-epoch (not per-iteration), and uses mini-batches
    so we don't blow up memory.
  - Tracks both per-iteration and per-epoch metrics.
  - Optional `augmentor` callable that mutates `(X, y)` per batch.
  - Optional `verbose` toggle.
  - Optional `eval_set_name` for nicer logging.
  - Calls model.train() / model.eval() so dropout / batchnorm behave correctly.
"""

import os
import time
import numpy as np
from tqdm import tqdm


def _batched_forward(model, X, batch_size=512):
    """Forward in mini-batches, concatenate outputs."""
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size]))
    return np.concatenate(out, axis=0)


class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn,
                 batch_size=128, scheduler=None,
                 augmentor=None, eval_batch_size=1024,
                 verbose=True, show_progress=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.augmentor = augmentor
        self.verbose = verbose
        self.show_progress = show_progress

        # Per-iteration history (loss only — score per-iteration is noisy/expensive).
        self.train_loss = []
        self.train_scores = []
        # Per-epoch history (full evaluation on dev set).
        self.epoch_train_loss = []
        self.epoch_train_score = []
        self.epoch_dev_loss = []
        self.epoch_dev_score = []
        self.lr_history = []

        self.best_score = 0.0

    # ----------------------------------------------------------------------
    def evaluate(self, data_set):
        X, y = data_set
        if hasattr(self.model, 'eval'):
            self.model.eval()
        logits = _batched_forward(self.model, X, self.eval_batch_size)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        if hasattr(self.model, 'train'):
            self.model.train()
        return float(score), float(loss)

    # ----------------------------------------------------------------------
    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get('num_epochs', 5)
        log_iters = kwargs.get('log_iters', 100)
        save_dir = kwargs.get('save_dir', 'best_model')
        save_name = kwargs.get('save_name', 'best_model.pickle')

        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self.model, 'train'):
            self.model.train()

        best_score = 0.0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(X.shape[0])
            X_shuf = X[idx]
            y_shuf = y[idx]

            n_iters = (X.shape[0] + self.batch_size - 1) // self.batch_size

            t0 = time.time()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            iterator = range(n_iters)
            if self.show_progress:
                iterator = tqdm(iterator,
                                desc=f'Epoch {epoch+1}/{num_epochs}',
                                leave=False)

            for it in iterator:
                start = it * self.batch_size
                end = min(start + self.batch_size, X.shape[0])
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                if self.augmentor is not None:
                    xb, yb = self.augmentor(xb, yb)

                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                self.train_loss.append(float(loss))

                # cheap online accuracy
                preds = np.argmax(logits, axis=-1)
                running_correct += int((preds == yb).sum())
                running_total += yb.shape[0]
                running_loss += float(loss) * yb.shape[0]
                self.train_scores.append(running_correct / max(1, running_total))

                # backward
                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.lr_history.append(self.optimizer.init_lr)

            # ------ end of epoch evaluation ------
            ep_train_loss = running_loss / max(1, running_total)
            ep_train_score = running_correct / max(1, running_total)
            ep_dev_score, ep_dev_loss = self.evaluate(dev_set)

            self.epoch_train_loss.append(ep_train_loss)
            self.epoch_train_score.append(ep_train_score)
            self.epoch_dev_loss.append(ep_dev_loss)
            self.epoch_dev_score.append(ep_dev_score)

            dt = time.time() - t0
            if self.verbose:
                print(f"[Epoch {epoch+1:>2}/{num_epochs}] "
                      f"train_loss={ep_train_loss:.4f} train_acc={ep_train_score:.4f} "
                      f"dev_loss={ep_dev_loss:.4f} dev_acc={ep_dev_score:.4f} "
                      f"lr={self.optimizer.init_lr:.2e}  ({dt:.1f}s)")

            if ep_dev_score > best_score:
                best_score = ep_dev_score
                save_path = os.path.join(save_dir, save_name)
                self.save_model(save_path)
                if self.verbose:
                    print(f"  -> new best {best_score:.4f}, saved to {save_path}")

        self.best_score = best_score
        return best_score

    # ----------------------------------------------------------------------
    def save_model(self, save_path):
        self.model.save_model(save_path)
