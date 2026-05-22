"""
lr_scheduler.py — learning-rate schedulers.

`scheduler.step()` is called once per training iteration (mini-batch).

Implements:
    - StepLR
    - MultiStepLR
    - ExponentialLR
    - CosineAnnealingLR
    - LinearWarmup     (wraps another scheduler)
"""

from abc import abstractmethod
import numpy as np


class scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.init_lr
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    """Decay lr by `gamma` every `step_size` iterations."""

    def __init__(self, optimizer, step_size=30, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.init_lr *= self.gamma


class MultiStepLR(scheduler):
    """Decay lr by `gamma` once the iteration count reaches each milestone."""

    def __init__(self, optimizer, milestones=None, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = sorted(milestones) if milestones is not None else []
        self.gamma = gamma
        self._milestones_set = set(self.milestones)

    def step(self):
        self.step_count += 1
        if self.step_count in self._milestones_set:
            self.optimizer.init_lr *= self.gamma


class ExponentialLR(scheduler):
    """Multiply lr by `gamma` every step (lr = base_lr * gamma**t)."""

    def __init__(self, optimizer, gamma=0.999):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma


class CosineAnnealingLR(scheduler):
    """Cosine annealing — lr decays smoothly from base_lr to eta_min over T_max iters."""

    def __init__(self, optimizer, T_max=1000, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self):
        self.step_count += 1
        t = min(self.step_count, self.T_max)
        new_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * t / self.T_max))
        self.optimizer.init_lr = float(new_lr)


class LinearWarmup(scheduler):
    """Wrap an underlying scheduler with a linear-warmup phase.

    For the first `warmup_iters` iterations, lr ramps linearly from
    `start_factor * base_lr` to `base_lr`. Afterwards, the inner scheduler
    takes over.
    """

    def __init__(self, optimizer, inner_scheduler, warmup_iters=200, start_factor=0.1):
        super().__init__(optimizer)
        self.inner = inner_scheduler
        self.warmup_iters = warmup_iters
        self.start_factor = start_factor
        # Save base lr at construction time, since inner may already mutate it.
        self.base_lr = optimizer.init_lr

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_iters:
            frac = self.step_count / max(1, self.warmup_iters)
            lr = (self.start_factor + (1.0 - self.start_factor) * frac) * self.base_lr
            self.optimizer.init_lr = float(lr)
        else:
            self.inner.step()
