"""
models.py — model classes built on the operators in `op.py`.

Contains:
    - Model_MLP : multi-layer perceptron with optional dropout / batchnorm
    - Model_CNN : LeNet-style convolutional model
"""

import pickle
from .op import (
    Layer, Linear, conv2D, MaxPool2D, Flatten,
    ReLU, LeakyReLU, Sigmoid, Tanh, ELU,
    Dropout, BatchNorm1d,
)


def _make_act(name):
    name = (name or 'ReLU').lower()
    if name in ('relu',):
        return ReLU()
    if name in ('leakyrelu', 'lrelu'):
        return LeakyReLU(0.01)
    if name in ('sigmoid', 'logistic'):
        return Sigmoid()
    if name in ('tanh',):
        return Tanh()
    if name in ('elu',):
        return ELU()
    raise ValueError(f"unknown activation: {name}")


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------

class Model_MLP(Layer):
    """Multi-layer perceptron.

    Parameters
    ----------
    size_list : list[int]
        Layer widths e.g. [784, 600, 10].
    act_func : str
        'ReLU' / 'LeakyReLU' / 'Sigmoid' / 'Tanh' / 'ELU'.
    lambda_list : list[float] or None
        Per-Linear-layer weight-decay coefficient. Same length as
        `len(size_list) - 1`.
    dropout : float
        Dropout probability inserted *between* hidden layers (after activation).
    batch_norm : bool
        If True, apply BatchNorm1d after each hidden Linear (before activation).
    init : str
        'he' | 'xavier' | 'normal' — weight initialization for Linear layers.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None,
                 dropout=0.0, batch_norm=False, init='he'):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.init = init

        self.layers = []
        if size_list is not None and act_func is not None:
            self._build(size_list, act_func, lambda_list, dropout, batch_norm, init)

    def _build(self, size_list, act_func, lambda_list, dropout, batch_norm, init):
        self.layers = []
        for i in range(len(size_list) - 1):
            in_dim = size_list[i]
            out_dim = size_list[i + 1]
            layer = Linear(in_dim=in_dim, out_dim=out_dim, initialize_method=init)
            if lambda_list is not None and lambda_list[i] is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[i]
            self.layers.append(layer)
            if i < len(size_list) - 2:  # hidden layer
                if batch_norm:
                    self.layers.append(BatchNorm1d(out_dim))
                self.layers.append(_make_act(act_func))
                if dropout and dropout > 0:
                    self.layers.append(Dropout(dropout))

    # ---------------- forward / backward ------------------
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None, 'Model not initialized.'
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        g = loss_grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    # ---------------- train / eval ------------------------
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    # ---------------- save / load -------------------------
    def save_model(self, save_path):
        meta = {
            'size_list': self.size_list,
            'act_func': self.act_func,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'init': self.init,
            'kind': 'MLP',
        }
        param_list = [meta]
        for layer in self.layers:
            if getattr(layer, 'optimizable', False) and isinstance(layer, Linear):
                param_list.append({
                    'type': 'Linear',
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
            elif isinstance(layer, BatchNorm1d):
                param_list.append({
                    'type': 'BatchNorm1d',
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'running_mean': layer.running_mean,
                    'running_var': layer.running_var,
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            param_list = pickle.load(f)
        # Backwards compatible: old format had [size_list, act_func, layer_dicts...]
        if isinstance(param_list[0], list):
            self.size_list = param_list[0]
            self.act_func = param_list[1]
            self.dropout = 0.0
            self.batch_norm = False
            self.init = 'he'
            self._build(self.size_list, self.act_func, None,
                        self.dropout, self.batch_norm, self.init)
            for i, layer in enumerate([l for l in self.layers if isinstance(l, Linear)]):
                d = param_list[2 + i]
                layer.params['W'] = d['W']
                layer.params['b'] = d['b']
                layer.W = d['W']
                layer.b = d['b']
                layer.weight_decay = d.get('weight_decay', False)
                layer.weight_decay_lambda = d.get('lambda', 1e-4)
            return

        meta = param_list[0]
        self.size_list = meta['size_list']
        self.act_func = meta['act_func']
        self.dropout = meta.get('dropout', 0.0)
        self.batch_norm = meta.get('batch_norm', False)
        self.init = meta.get('init', 'he')
        self._build(self.size_list, self.act_func, None,
                    self.dropout, self.batch_norm, self.init)

        ptr = 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                d = param_list[ptr]; ptr += 1
                layer.params['W'] = d['W']
                layer.params['b'] = d['b']
                layer.W = d['W']
                layer.b = d['b']
                layer.weight_decay = d.get('weight_decay', False)
                layer.weight_decay_lambda = d.get('lambda', 1e-4)
            elif isinstance(layer, BatchNorm1d):
                d = param_list[ptr]; ptr += 1
                layer.params['W'] = d['W']
                layer.params['b'] = d['b']
                layer.gamma = d['W']
                layer.beta = d['b']
                layer.running_mean = d['running_mean']
                layer.running_var = d['running_var']


# ----------------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------------

class Model_CNN(Layer):
    """A LeNet-style convolutional network.

    Default architecture (input shape [N, 1, 28, 28]):
        Conv(1->8, 3x3, p=1) -> ReLU -> MaxPool 2x2
        Conv(8->16, 3x3, p=1) -> ReLU -> MaxPool 2x2
        Flatten -> Linear(16*7*7 -> 128) -> ReLU [-> Dropout]
        Linear(128 -> 10)

    Parameters
    ----------
    in_channels : int
    conv_channels : list[int]
        e.g. [8, 16] gives two conv layers with 8 and 16 output channels.
    kernel_size : int
    fc_sizes : list[int]
        Sizes of FC hidden layers after flatten (output 10 is appended).
    input_hw : tuple
        Spatial size at input (height, width).
    num_classes : int
    weight_decay_lambda : float or None
    dropout : float
    init : str
    """

    def __init__(self,
                 in_channels=1,
                 conv_channels=(8, 16),
                 kernel_size=3,
                 fc_sizes=(128,),
                 input_hw=(28, 28),
                 num_classes=10,
                 weight_decay_lambda=None,
                 dropout=0.0,
                 init='he'):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.fc_sizes = list(fc_sizes)
        self.input_hw = tuple(input_hw)
        self.num_classes = num_classes
        self.weight_decay_lambda = weight_decay_lambda
        self.dropout = dropout
        self.init = init

        self.layers = []
        self._build()

    def _build(self):
        self.layers = []
        c_prev = self.in_channels
        h, w = self.input_hw
        for c in self.conv_channels:
            conv = conv2D(
                in_channels=c_prev, out_channels=c,
                kernel_size=self.kernel_size,
                stride=1, padding=self.kernel_size // 2,
                initialize_method=self.init,
                weight_decay=self.weight_decay_lambda is not None,
                weight_decay_lambda=self.weight_decay_lambda or 0.0,
            )
            self.layers.append(conv)
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(2, 2))
            c_prev = c
            h, w = h // 2, w // 2

        self.layers.append(Flatten())

        fc_in = c_prev * h * w
        for fc in self.fc_sizes:
            lin = Linear(
                in_dim=fc_in, out_dim=fc,
                initialize_method=self.init,
                weight_decay=self.weight_decay_lambda is not None,
                weight_decay_lambda=self.weight_decay_lambda or 0.0,
            )
            self.layers.append(lin)
            self.layers.append(ReLU())
            if self.dropout and self.dropout > 0:
                self.layers.append(Dropout(self.dropout))
            fc_in = fc

        # Final classifier
        self.layers.append(Linear(
            in_dim=fc_in, out_dim=self.num_classes,
            initialize_method=self.init,
            weight_decay=self.weight_decay_lambda is not None,
            weight_decay_lambda=self.weight_decay_lambda or 0.0,
        ))

    # ----- forward / backward ---------------------------------
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # If input is flattened, reshape on the fly.
        if X.ndim == 2:
            X = X.reshape(-1, self.in_channels, *self.input_hw)
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        g = loss_grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    # ----- save / load ----------------------------------------
    def save_model(self, save_path):
        meta = {
            'kind': 'CNN',
            'in_channels': self.in_channels,
            'conv_channels': self.conv_channels,
            'kernel_size': self.kernel_size,
            'fc_sizes': self.fc_sizes,
            'input_hw': self.input_hw,
            'num_classes': self.num_classes,
            'weight_decay_lambda': self.weight_decay_lambda,
            'dropout': self.dropout,
            'init': self.init,
        }
        params = [meta]
        for layer in self.layers:
            if isinstance(layer, (Linear, conv2D)):
                params.append({
                    'type': layer.__class__.__name__,
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'lambda': getattr(layer, 'weight_decay_lambda', 0.0),
                })
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        meta = params[0]
        self.in_channels = meta['in_channels']
        self.conv_channels = meta['conv_channels']
        self.kernel_size = meta['kernel_size']
        self.fc_sizes = meta['fc_sizes']
        self.input_hw = tuple(meta['input_hw'])
        self.num_classes = meta['num_classes']
        self.weight_decay_lambda = meta['weight_decay_lambda']
        self.dropout = meta['dropout']
        self.init = meta['init']
        self._build()

        ptr = 1
        for layer in self.layers:
            if isinstance(layer, (Linear, conv2D)):
                d = params[ptr]; ptr += 1
                layer.params['W'] = d['W']
                layer.params['b'] = d['b']
                layer.W = d['W']
                layer.b = d['b']
                layer.weight_decay = d.get('weight_decay', False)
                layer.weight_decay_lambda = d.get('lambda', 0.0)
