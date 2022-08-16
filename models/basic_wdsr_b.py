from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['BASIC_MODEL', ]

Conv2d = nn.Conv2d


class BASIC_MODEL(nn.Module):

    def __init__(self, params):
        super(BASIC_MODEL, self).__init__()
        self.image_mean = params.image_mean
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = params.num_channels
        scale = params.scale
        self.scale = scale
        self.remain_blocks = params.num_blocks
        self.kwargs = {}

        num_outputs = scale * scale * params.num_channels

        conv = weight_norm(
            Conv2d(
                num_inputs,
                params.num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
                **self.kwargs))

        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.head = conv
        body = nn.ModuleList()
        for _ in range(params.num_blocks):
            body.append(
                Block(
                    num_residual_units=params.num_residual_units,
                    kernel_size=kernel_size,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(params.num_blocks),
                    **self.kwargs,
                )
            )
        self.body = body
        conv = weight_norm(
            Conv2d(
                params.num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2,
                **self.kwargs))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.tail = conv

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(
                Conv2d(
                    num_inputs,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2,
                    **self.kwargs))
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        x = x - self.image_mean
        y = self.head(x)
        for module in self.body:
            y = module(y)
        x = self.tail(y) + self.skip(x)
        x = self.shuf(x)
        x = x + self.image_mean
        return x


class Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1,
                 **kwargs):
        super(Block, self).__init__()
        body = []
        expand = 6
        linear = 0.84
        conv = weight_norm(
            Conv2d(
                num_residual_units,
                int(num_residual_units * expand),
                1,
                padding=1 // 2,
                **kwargs))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(inplace=True))
        conv = weight_norm(
            Conv2d(
                num_residual_units * expand,
                int(num_residual_units * linear),
                1,
                padding=1 // 2,
                **kwargs))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        conv = weight_norm(
            Conv2d(
                int(num_residual_units * linear),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2,
                **kwargs))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x
