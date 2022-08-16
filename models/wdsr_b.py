from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.ops import BinaryConv2d, rounding

try:
    from speed_models import PseudoLength, BlockBSpeedEstimator
except ImportError:
    pass

from collections import namedtuple

ModelOutput = namedtuple(
    "ModelOutput",
    "sr speed_accu speed_curr"
)

__all__ = ['NAS_MODEL', ]


class NAS_MODEL(nn.Module):

    def __init__(self, params):
        super(NAS_MODEL, self).__init__()
        self.image_mean = params.image_mean
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = params.num_channels
        scale = params.scale
        self.scale = scale
        self.num_blocks = params.num_blocks
        self.num_residual_units = params.num_residual_units
        self.remain_blocks = params.num_blocks
        self.width_search = params.width_search

        num_outputs = scale * scale * params.num_channels

        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                params.num_residual_units,
                kernel_size,
                padding=kernel_size // 2))

        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.head = conv
        self.speed_estimator = BlockBSpeedEstimator('mask' if params.width_search else 'channel').eval()

        body = nn.ModuleList()
        for _ in range(params.num_blocks):
            body.append(AggregationLayer(
                num_residual_units=params.num_residual_units,
                kernel_size=kernel_size,
                weight_norm=weight_norm,
                res_scale=1 / math.sqrt(params.num_blocks),
                width_search=params.width_search)
            )
        self.body = body

        if self.width_search:
            self.mask = BinaryConv2d(in_channels=params.num_residual_units,
                                          out_channels=params.num_residual_units,
                                          groups=params.num_residual_units)
            # self.mask.init(0.5)
        conv = weight_norm(
            nn.Conv2d(
                params.num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.tail = conv

        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_outputs,
                skip_kernel_size,
                padding=skip_kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.skip = conv

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

        if params.pretrained:
            self.load_pretrained()

    def forward(self, x):
        x = x - self.image_mean
        y = self.head(x)
        speed_accu = x.new_zeros(1)
        for module in self.body:
            if self.width_search:
                speed_curr = self.speed_estimator.estimateByMask(module, self.mask)
            else:
                speed_curr = self.speed_estimator.estimateByChannelNum(module)
            y = self.mask(y)
            y, speed_accu = module(y, speed_curr, speed_accu)
        if self.width_search:
            y = self.mask(y)
        y = self.tail(y) + self.skip(x)
        y = self.shuf(y)
        y = y + self.image_mean
        return y, speed_accu

    @torch.no_grad()
    def get_current_blocks(self):
        num_blocks = 0
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                if module.alpha1 < module.alpha2:  # not skip
                    num_blocks += 1
        return int(num_blocks)

    @torch.no_grad()
    def get_block_status(self):
        remain_block_idx = []
        for idx, module in enumerate(self.body.children()):
            if isinstance(module, AggregationLayer):
                alpha1, alpha2 = F.softmax(torch.stack([module.alpha1, module.alpha2], dim=0), dim=0)
                if alpha1 < alpha2:  # not skip
                    remain_block_idx.append(idx)
        return remain_block_idx

    @torch.no_grad()
    def get_width_from_block_idx(self, remain_block_idx):
        @torch.no_grad()
        def _get_width_from_weight(w):
            return int(rounding(w).sum())

        all_width = []
        for idx, module in enumerate(self.body.children()):
            width = []
            if idx in remain_block_idx and isinstance(module, AggregationLayer):
                # width.append(_get_width_from_weight(module.block_mask.weight))
                width.append(_get_width_from_weight(self.mask.weight))
                for m in module.body.children():
                    if isinstance(m, BinaryConv2d):
                        width.append(_get_width_from_weight(m.weight))
                all_width.append(width)
        return all_width

    @torch.no_grad()
    def get_alpha_grad(self):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                return module.alpha1.grad, module.alpha2.grad

    @torch.no_grad()
    def get_alpha(self):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                return module.alpha1, module.alpha2

    @torch.no_grad()
    def length_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                module.alpha1.requires_grad = flag
                module.alpha2.requires_grad = flag
                module.beta1.requires_grad = flag
                module.beta2.requires_grad = flag

    @torch.no_grad()
    def mask_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                # module.block_mask.weight.requires_grad = flag
                for m in module.body.children():
                    if isinstance(m, BinaryConv2d):
                        m.weight.requires_grad = flag
        self.mask.weight.requires_grad = flag

    @torch.no_grad()
    def get_mask_grad(self):
        return self.mask.weight.grad

    @torch.no_grad()
    def get_mask_weight(self):
        return self.mask.weight.data

    @torch.no_grad()
    def load_pretrained(self):
        import os
        path, filename = os.path.split(__file__)
        weight_path = f"{path}/pretrained_weights"
        state_dict = torch.load(f"{weight_path}/wdsr_b_x{self.scale}_{self.num_blocks}_{self.num_residual_units}.pt",
                                map_location='cpu')
        state_dict_iterator = iter(state_dict.items())
        load_name, load_param = next(state_dict_iterator)
        for p in self.parameters():
            if p.size() == load_param.size():
                p.data = load_param
                try:
                    load_name, load_param = next(state_dict_iterator)
                except StopIteration:
                    pass


class Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1,
                 width_search=False):
        super(Block, self).__init__()
        body = []
        expand = 6
        linear = 0.84

        # if width_search:
        #     # mask for first layer
        #     self.block_mask = BinaryConv2d(in_channels=num_residual_units,
        #                                    out_channels=num_residual_units,
        #                                    groups=num_residual_units)
        #     self.block_mask.init(0.5)
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * expand),
                1,
                padding=1 // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)

        body.append(conv)
        body.append(nn.ReLU(inplace=True))
        if width_search:
            # mask for second layer
            body.append(BinaryConv2d(in_channels=int(num_residual_units * expand),
                                     out_channels=int(num_residual_units * expand),
                                     groups=int(num_residual_units * expand)))

        conv = weight_norm(
            nn.Conv2d(
                num_residual_units * expand,
                int(num_residual_units * linear),
                1,
                padding=1 // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)

        if width_search:
            # mask for third layer
            body.append(BinaryConv2d(in_channels=int(num_residual_units * linear),
                                     out_channels=int(num_residual_units * linear),
                                     groups=int(num_residual_units * linear)))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * linear),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # x = self.block_mask(x)
        x = self.body(x) + x
        return x


class AggregationLayer(Block):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

        # Skip
        self.alpha1 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta1 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        init.uniform_(self.alpha1, 0, 0.2)

        # Preserve
        self.alpha2 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta2 = nn.Parameter(data=torch.ones(1), requires_grad=True)
        init.uniform_(self.alpha2, 0.8, 1)

    def forward(self, x, speed_curr, speed_accu):
        # model_output = input
        # x = input.sr
        # speed_accu = input.speed_accu
        #
        # speed_curr = input.speed_curr
        if self.training:
            # self.alpha1.data, self.alpha2.data = F.gumbel_softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                                   dim=0), dim=0, hard=False)
            # self.alpha1.data, self.alpha2.data = F.softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                            dim=0), dim=0)
            # Get skip result
            sr1 = x
            # Get block result
            sr2 = self.body(x) + x

            beta1, beta2 = ConditionFunction.apply(self.alpha1, self.alpha2, self.beta1, self.beta2)
            self.beta1.data, self.beta2.data = beta1, beta2
            x = beta1 * sr1 + beta2 * sr2
            # model_output.speed_accu = beta2 * speed_curr + speed_accu
            speed_accu = beta2 * speed_curr + speed_accu
            return x, speed_accu
        else:
            if self.alpha1 >= self.alpha2:
                pass
            else:
                x = self.body(x) + x
            # model_output.speed_accu = speed_accu + self.beta2 * speed_curr
            speed_accu = speed_accu + self.beta2 * speed_curr
            return x, speed_accu

    def get_num_channels(self):
        channels = []
        for m in self.body.children():
            if isinstance(m, nn.Conv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        return channels


class AggregationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sr1, sr2, speed_accu, speed_curr, alpha1, alpha2, beta1, beta2):
        ctx.num_outputs = 2
        with torch.no_grad():
            if alpha1 > alpha2:
                # when a1 >= a2
                beta1.data = beta1.new_ones(1)  # set b1 = 0
                beta2.data = beta2.new_zeros(1)  # set b2 = 1
            else:
                # when a1 < a2
                beta1.data = beta1.new_zeros(1)  # set b1 = 0
                beta2.data = beta2.new_ones(1)  # set b2 = 1

        ctx.save_for_backward(sr1, sr2, speed_accu, speed_curr, beta1, beta2)

        sr = sr1 * beta1 + sr2 * beta2
        total_speed = speed_curr * beta2 + speed_accu

        return sr, total_speed

    @staticmethod
    def backward(ctx, grad_output_sr, grad_output_speed):
        sr1, sr2, speed_accu, speed_curr, beta1, beta2 = ctx.saved_tensors
        grad_sr1 = grad_output_sr * beta1
        grad_sr2 = grad_output_sr * beta2
        grad_speed_accu = grad_output_speed * beta2
        grad_speed_curr = grad_output_speed
        grad_beta1 = grad_output_sr.bmm(beta1)  # for grad_alpha1
        grad_beta2 = grad_output_sr * beta2 + grad_output_speed * speed_curr  # for grad_alpha2

        grad_alpha1 = grad_beta1
        grad_alpha2 = grad_beta2

        return grad_sr1, grad_sr2, grad_speed_accu, grad_speed_curr, grad_alpha1, grad_alpha2, None, None


class ConditionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha1, alpha2, beta1, beta2):
        with torch.no_grad():
            if alpha1 >= alpha2:
                # when a1 >= a2
                beta1.data = beta1.new_ones(1)  # set b1 = 0
                beta2.data = beta2.new_zeros(1)  # set b2 = 1
            else:
                # when a1 < a2
                beta1.data = beta1.new_zeros(1)  # set b1 = 0
                beta2.data = beta2.new_ones(1)  # set b2 = 1

        return beta1, beta2

    @staticmethod
    def backward(ctx, grad_output_beta1, grad_output_beta2):

        grad_alpha1 = grad_output_beta1
        grad_alpha2 = grad_output_beta2

        return grad_alpha1, grad_alpha2, None, None


if __name__ == "__main__":
    sr1 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    sr2 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    speed_accu = torch.rand(1, requires_grad=True, dtype=torch.float64)
    speed_curr = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a2 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b2 = torch.rand(1, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(ConditionFunction.apply, (a1, a2, b1, b2,), eps=1e-1)

    pass
