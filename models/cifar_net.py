"""Configurable CIFAR-10 CNN for Project-2 Task 1."""
from torch import nn
import torch


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, channels, activation="relu", use_batchnorm=True, dropout_p=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(get_activation(activation))
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", use_batchnorm=True,
                 dropout_p=0.0, use_residual=False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        if use_residual:
            layers.append(ResidualBlock(out_channels, activation, use_batchnorm, dropout_p))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class ConfigurableCIFARNet(nn.Module):
    """CNN containing Conv2d, pooling, activation, FC and optional BN/Dropout/Residual."""

    def __init__(self, filters=(64, 128, 256), activation="relu", use_batchnorm=True,
                 dropout_p=0.3, use_residual=False, num_classes=10):
        super().__init__()
        if len(filters) < 2:
            raise ValueError("filters must contain at least two channel sizes")

        in_channels = 3
        stages = []
        for out_channels in filters:
            stages.append(ConvStage(in_channels, out_channels, activation, use_batchnorm,
                                    dropout_p, use_residual))
            in_channels = out_channels
        self.features = nn.Sequential(*stages)

        spatial = 32 // (2 ** len(filters))
        if spatial < 1:
            raise ValueError("Too many pooling stages for 32x32 CIFAR-10 input")
        hidden = max(128, filters[-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1] * spatial * spatial, hidden),
            get_activation(activation),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parse_filters(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())
