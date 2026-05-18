from __future__ import annotations

import functools

import torch
from torch import nn


def init_weights(module: nn.Module, init_gain: float = 0.02) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, 0.0, init_gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm2d" in classname:
        nn.init.normal_(module.weight.data, 1.0, init_gain)
        nn.init.constant_(module.bias.data, 0.0)


def norm_layer():
    return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        norm = norm_layer()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            norm(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """ResNet generator used by CycleGAN."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        ngf: int = 64,
        num_blocks: int = 9,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        norm = norm_layer()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0, bias=True),
            norm(ngf),
            nn.ReLU(True),
        ]

        channels = ngf
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    norm(channels * 2),
                    nn.ReLU(True),
                ]
            )
            channels *= 2

        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))

        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        channels // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=True,
                    ),
                    norm(channels // 2),
                    nn.ReLU(True),
                ]
            )
            channels //= 2

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels, output_channels, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, input_channels: int = 3, ndf: int = 64, num_layers: int = 3):
        super().__init__()
        norm = norm_layer()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        channels = ndf
        for layer_index in range(1, num_layers):
            next_channels = min(ndf * 2**layer_index, ndf * 8)
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    norm(next_channels),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            channels = next_channels

        next_channels = min(channels * 2, ndf * 8)
        layers.extend(
            [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1, bias=True),
                norm(next_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

