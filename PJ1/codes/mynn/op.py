"""
op.py — core operators for the NumPy-only neural network framework.

Implements:
    - Linear  (with He / Xavier / Normal initialization, optional weight decay)
    - conv2D  (im2col-based, supports stride / padding)
    - MaxPool2D
    - Flatten
    - Activations: ReLU, LeakyReLU, Sigmoid, Tanh, ELU
    - Dropout
    - BatchNorm1d
    - MultiCrossEntropyLoss (with built-in numerically stable softmax)
    - L2Regularization (helper)
"""

from abc import abstractmethod
import numpy as np


# ----------------------------------------------------------------------------
# Initializers
# ----------------------------------------------------------------------------

def _he_normal(size):
    fan_in = size[0] if len(size) == 2 else int(np.prod(size[1:]))
    return np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=size)


def _xavier_normal(size):
    fan_in = size[0] if len(size) == 2 else int(np.prod(size[1:]))
    fan_out = size[1] if len(size) == 2 else size[0]
    return np.random.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=size)


def _small_normal(size):
    return np.random.normal(0.0, 0.01, size=size)


INIT_METHODS = {
    'he': _he_normal,
    'kaiming': _he_normal,
    'xavier': _xavier_normal,
    'normal': _small_normal,
    'default': np.random.normal,
}


def get_init(name_or_callable):
    if callable(name_or_callable):
        return name_or_callable
    return INIT_METHODS.get(name_or_callable, _he_normal)


# ----------------------------------------------------------------------------
# Base layer
# ----------------------------------------------------------------------------

class Layer:
    def __init__(self):
        self.optimizable = True
        self.training = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# ----------------------------------------------------------------------------
# Linear
# ----------------------------------------------------------------------------

class Linear(Layer):
    """Fully-connected layer.

    Parameters
    ----------
    in_dim, out_dim : int
    initialize_method : 'he' | 'xavier' | 'normal' | callable
    weight_decay : bool
        If True, an L2 weight-decay term is included by the optimizer.
    weight_decay_lambda : float
        Strength of the weight-decay term.
    """

    def __init__(self, in_dim, out_dim,
                 initialize_method='he',
                 weight_decay=False, weight_decay_lambda=1e-4):
        super().__init__()
        init = get_init(initialize_method)
        self.W = init(size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.grads = {'W': None, 'b': None}
        self.input = None

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """X: [B, in_dim]  ->  [B, out_dim]"""
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad):
        """grad: [B, out_dim]  ->  returns dX [B, in_dim]

        The loss already supplies the (1/B) factor in its gradient, so we do
        NOT average again here.
        """
        # gradients
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] = self.grads['W'] + self.weight_decay_lambda * self.params['W']
        # propagate
        dX = grad @ self.params['W'].T
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


# ----------------------------------------------------------------------------
# conv2D  (im2col implementation)
# ----------------------------------------------------------------------------

def _im2col(x, kh, kw, stride, padding):
    """
    x : [N, C, H, W]
    returns : cols [N * out_h * out_w, C * kh * kw], (out_h, out_w)
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    if padding > 0:
        x = np.pad(x,
                   ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                   mode='constant')

    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, (out_h, out_w)


def _col2im(cols, x_shape, kh, kw, stride, padding):
    """Inverse of _im2col — reduces with addition for overlapping receptive fields."""
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    Hp = H + 2 * padding
    Wp = W + 2 * padding
    img = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)

    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img


class conv2D(Layer):
    """2-D convolutional layer (NCHW)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 initialize_method='he',
                 weight_decay=False, weight_decay_lambda=1e-4):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = kh
        self.kw = kw
        self.stride = stride
        self.padding = padding

        init = get_init(initialize_method)
        # Shape [out_channels, in_channels, kh, kw]
        self.W = init(size=(out_channels, in_channels, kh, kw)).astype(np.float32)
        self.b = np.zeros((out_channels,), dtype=np.float32)

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self._cache = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """X : [N, C_in, H, W] -> [N, C_out, oh, ow]"""
        N, C, H, W = X.shape
        cols, (oh, ow) = _im2col(X, self.kh, self.kw, self.stride, self.padding)
        # Filter matrix [C_out, C_in*kh*kw]
        Wcol = self.params['W'].reshape(self.out_channels, -1)
        out = cols @ Wcol.T + self.params['b']  # [N*oh*ow, C_out]
        out = out.reshape(N, oh, ow, self.out_channels).transpose(0, 3, 1, 2)
        self._cache = (X.shape, cols, oh, ow)
        return out

    def backward(self, grad):
        """grad: [N, C_out, oh, ow]

        Loss already provides (1/B); we don't divide again.
        """
        x_shape, cols, oh, ow = self._cache
        # [N*oh*ow, C_out]
        grad_flat = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        Wcol = self.params['W'].reshape(self.out_channels, -1)
        # Gradients
        dW = grad_flat.T @ cols
        dW = dW.reshape(self.params['W'].shape)
        db = np.sum(grad_flat, axis=0)
        if self.weight_decay:
            dW = dW + self.weight_decay_lambda * self.params['W']
        self.grads['W'] = dW
        self.grads['b'] = db

        # dCols
        dcols = grad_flat @ Wcol  # [N*oh*ow, C_in*kh*kw]
        dx = _col2im(dcols, x_shape, self.kh, self.kw, self.stride, self.padding)
        return dx

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


# ----------------------------------------------------------------------------
# MaxPool2D
# ----------------------------------------------------------------------------

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.optimizable = False
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            self.kh, self.kw = kernel_size
        self.stride = stride if stride is not None else self.kh
        self._cache = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        oh = (H - self.kh) // self.stride + 1
        ow = (W - self.kw) // self.stride + 1

        # reshape to (N, C, oh, kh, ow, kw)
        x_strided = np.lib.stride_tricks.as_strided(
            X,
            shape=(N, C, oh, ow, self.kh, self.kw),
            strides=(X.strides[0], X.strides[1],
                     X.strides[2] * self.stride, X.strides[3] * self.stride,
                     X.strides[2], X.strides[3]),
            writeable=False,
        )
        x_block = x_strided.reshape(N, C, oh, ow, -1)
        argmax = np.argmax(x_block, axis=-1)
        out = np.max(x_block, axis=-1)
        self._cache = (X.shape, argmax, oh, ow)
        return out

    def backward(self, grad):
        x_shape, argmax, oh, ow = self._cache
        N, C, H, W = x_shape
        dx = np.zeros(x_shape, dtype=grad.dtype)
        # scatter
        for i in range(oh):
            for j in range(ow):
                idx = argmax[:, :, i, j]
                ki = idx // self.kw
                kj = idx % self.kw
                # write
                ni, ci = np.indices((N, C))
                dx[ni, ci, i * self.stride + ki, j * self.stride + kj] += grad[:, :, i, j]
        return dx


# ----------------------------------------------------------------------------
# Flatten
# ----------------------------------------------------------------------------

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self._shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self._shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)


# ----------------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------------

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.maximum(X, 0)

    def backward(self, grad):
        return np.where(self.input > 0, grad, 0)


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.optimizable = False
        self.alpha = alpha
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, grad):
        return np.where(self.input > 0, grad, self.alpha * grad)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.out = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.out = 1.0 / (1.0 + np.exp(-X))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.out = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class ELU(Layer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.optimizable = False
        self.alpha = alpha
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * (np.exp(np.minimum(X, 0)) - 1))

    def backward(self, grad):
        return np.where(self.input > 0,
                        grad,
                        grad * self.alpha * np.exp(np.minimum(self.input, 0)))


# ----------------------------------------------------------------------------
# Dropout
# ----------------------------------------------------------------------------

class Dropout(Layer):
    """Inverted dropout: scale by 1/(1-p) during training, identity at eval."""

    def __init__(self, p=0.5):
        super().__init__()
        self.optimizable = False
        self.p = p
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training and self.p > 0:
            self.mask = (np.random.rand(*X.shape) >= self.p).astype(X.dtype) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, grad):
        if self.training and self.p > 0:
            return grad * self.mask
        return grad


# ----------------------------------------------------------------------------
# BatchNorm1d
# ----------------------------------------------------------------------------

class BatchNorm1d(Layer):
    """Batch normalization for fully-connected layers.

    Maintains running mean/var so eval mode uses fixed statistics.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, num_features), dtype=np.float32)
        self.beta = np.zeros((1, num_features), dtype=np.float32)
        self.params = {'W': self.gamma, 'b': self.beta}
        self.grads = {'W': None, 'b': None}

        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

        self.weight_decay = False
        self.weight_decay_lambda = 0.0

        self._cache = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_centered = X - mean
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_centered * std_inv
        out = self.params['W'] * x_hat + self.params['b']
        self._cache = (x_hat, x_centered, std_inv)
        return out

    def backward(self, grad):
        x_hat, x_centered, std_inv = self._cache
        B = grad.shape[0]

        self.grads['W'] = np.sum(grad * x_hat, axis=0, keepdims=True)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        dxhat = grad * self.params['W']
        dvar = np.sum(dxhat * x_centered, axis=0, keepdims=True) * -0.5 * std_inv ** 3
        dmean = np.sum(dxhat * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * x_centered, axis=0, keepdims=True)
        dx = dxhat * std_inv + dvar * 2.0 * x_centered / B + dmean / B
        return dx

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class MultiCrossEntropyLoss(Layer):
    """Numerically stable softmax + cross entropy.

    Supports `cancel_soft_max()` if you want to feed already-softmaxed inputs.
    """

    def __init__(self, model=None, max_classes=10):
        super().__init__()
        self.optimizable = False
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        labels = labels.astype(np.int64)
        self.labels = labels
        if self.has_softmax:
            # log-sum-exp trick for stable log-softmax
            x_max = np.max(predicts, axis=1, keepdims=True)
            shifted = predicts - x_max
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            log_probs = shifted - log_sum_exp
            self.predicts = np.exp(log_probs)  # softmax probabilities
            B = predicts.shape[0]
            nll = -log_probs[np.arange(B), labels]
            return float(np.mean(nll))
        else:
            # predicts are already probabilities
            self.predicts = predicts
            B = predicts.shape[0]
            eps = 1e-12
            nll = -np.log(predicts[np.arange(B), labels] + eps)
            return float(np.mean(nll))

    def backward(self):
        B = self.predicts.shape[0]
        if self.has_softmax:
            grads = self.predicts.copy()
            grads[np.arange(B), self.labels] -= 1.0
            grads = grads / B
        else:
            grads = np.zeros_like(self.predicts)
            grads[np.arange(B), self.labels] = -1.0 / (self.predicts[np.arange(B), self.labels] + 1e-12)
            grads = grads / B
        self.grads = grads
        if self.model is not None:
            self.model.backward(grads)
        return grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


# ----------------------------------------------------------------------------
# L2 Regularization helper (for bookkeeping only — actual decay applied via
# `weight_decay` flag on each layer or via AdamW).
# ----------------------------------------------------------------------------

class L2Regularization:
    """Compute L2 regularization loss across a model — used for reporting only."""

    def __init__(self, model):
        self.model = model

    def __call__(self):
        total = 0.0
        for layer in self.model.layers:
            if getattr(layer, 'weight_decay', False) and 'W' in getattr(layer, 'params', {}):
                W = layer.params['W']
                total += 0.5 * layer.weight_decay_lambda * float(np.sum(W * W))
        return total
