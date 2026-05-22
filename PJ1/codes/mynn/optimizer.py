"""
optimizer.py — optimizers for the NumPy framework.

Implements:
    - SGD
    - MomentGD          (vanilla momentum, with optional Nesterov)
    - Adam              (Kingma & Ba, 2014)
    - AdamW             (Loshchilov & Hutter, 2019 — decoupled weight decay)
"""

from abc import abstractmethod
import numpy as np


# ----------------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------------

class Optimizer:
    def __init__(self, init_lr, model):
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

    # --- helpers --------------------------------------------------------------
    def _iter_optimizable_params(self):
        """Yield (layer_id, key, layer) over all optimizable params."""
        for li, layer in enumerate(self.model.layers):
            if not getattr(layer, 'optimizable', False):
                continue
            if not hasattr(layer, 'params') or not hasattr(layer, 'grads'):
                continue
            for key in layer.params.keys():
                yield li, key, layer


# ----------------------------------------------------------------------------
# Plain SGD (with optional weight decay handled by Linear/Conv layers)
# ----------------------------------------------------------------------------

class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if not getattr(layer, 'optimizable', False):
                continue
            for key in layer.params.keys():
                if layer.grads.get(key) is None:
                    continue
                layer.params[key] -= self.init_lr * layer.grads[key]


# ----------------------------------------------------------------------------
# Momentum / Nesterov
# ----------------------------------------------------------------------------

class MomentGD(Optimizer):
    """Stochastic gradient descent with momentum.

    Parameters
    ----------
    init_lr : float
    model : nn model
    mu : float
        Momentum coefficient (commonly 0.9).
    nesterov : bool
        If True, use Nesterov-accelerated gradient.
    """

    def __init__(self, init_lr, model, mu=0.9, nesterov=False):
        super().__init__(init_lr, model)
        self.mu = mu
        self.nesterov = nesterov
        self.velocity = {}
        for li, key, layer in self._iter_optimizable_params():
            self.velocity[(li, key)] = np.zeros_like(layer.params[key])

    def step(self):
        for li, key, layer in self._iter_optimizable_params():
            g = layer.grads.get(key)
            if g is None:
                continue
            v_prev = self.velocity[(li, key)]
            v_new = self.mu * v_prev - self.init_lr * g
            self.velocity[(li, key)] = v_new
            if self.nesterov:
                layer.params[key] += -self.mu * v_prev + (1 + self.mu) * v_new
            else:
                layer.params[key] += v_new


# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------

class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
        for li, key, layer in self._iter_optimizable_params():
            self.m[(li, key)] = np.zeros_like(layer.params[key])
            self.v[(li, key)] = np.zeros_like(layer.params[key])

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for li, key, layer in self._iter_optimizable_params():
            g = layer.grads.get(key)
            if g is None:
                continue
            m = self.beta1 * self.m[(li, key)] + (1 - self.beta1) * g
            v = self.beta2 * self.v[(li, key)] + (1 - self.beta2) * (g * g)
            self.m[(li, key)] = m
            self.v[(li, key)] = v
            m_hat = m / bc1
            v_hat = v / bc2
            layer.params[key] -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------------------
# AdamW
# ----------------------------------------------------------------------------

class AdamW(Optimizer):
    """AdamW — decoupled weight decay.

    The decay term is applied directly to the parameters rather than mixed
    into the running estimate of the gradient. Decay strength is read from
    `layer.weight_decay_lambda` whenever `layer.weight_decay` is True.
    """

    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
        for li, key, layer in self._iter_optimizable_params():
            self.m[(li, key)] = np.zeros_like(layer.params[key])
            self.v[(li, key)] = np.zeros_like(layer.params[key])

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for li, key, layer in self._iter_optimizable_params():
            g = layer.grads.get(key)
            if g is None:
                continue
            # Adam update on raw gradient (without weight decay mixed in)
            m = self.beta1 * self.m[(li, key)] + (1 - self.beta1) * g
            v = self.beta2 * self.v[(li, key)] + (1 - self.beta2) * (g * g)
            self.m[(li, key)] = m
            self.v[(li, key)] = v
            m_hat = m / bc1
            v_hat = v / bc2
            update = self.init_lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Decoupled weight decay (only on weights, not biases)
            if getattr(layer, 'weight_decay', False) and key == 'W':
                update += self.init_lr * layer.weight_decay_lambda * layer.params[key]

            layer.params[key] -= update
