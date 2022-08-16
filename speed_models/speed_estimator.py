from .SpeedModel import block_b
import torch
import torch.nn as nn
from models.ops import BinaryConv2d, rounding
from loss_config import mobile_device, compute_device


class BlockBSpeedEstimator(nn.Module):

    def __init__(self, type, mobile_device=mobile_device, compute_device=compute_device, scale=2):
        super(BlockBSpeedEstimator, self).__init__()
        self.estimator = block_b(mobile_device, compute_device, scale).eval()  # load pretrained model
        self.type = type

    def forward(self, x):
        if self.type == 'channel':
            return self.estimateByModuleChannel(x.body)
        elif self.type == 'tensor':
            return self.estimateByChannelNum(x)
        elif self.type == 'mask':
            return self.estimateByMask(x)

    @torch.no_grad()
    def estimateByModuleChannel(self, module: nn.Module):
        channels = []
        for m in module.children():
            if isinstance(m, nn.Conv2d) and not isinstance(m, BinaryConv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        input_channels = torch.tensor(channels, dtype=torch.float).cuda()
        output = self.estimator(input_channels)
        return output

    @torch.no_grad()
    def estimateByChannelNum(self, x):
        output = self.estimator(x)
        return output

    @torch.no_grad()
    def estimateByMask(self, module: nn.Module, block_mask: nn.Module):
        # channels = self.get_unmask_number(module.block_mask)
        channels = self.get_unmask_number(block_mask)
        for m in module.body.children():
            if isinstance(m, BinaryConv2d):
                channels = torch.cat([channels, self.get_unmask_number(m)])
        channels = torch.cat([channels, channels[0].unsqueeze(0)])
        output = self.estimator(channels)
        return output

    @staticmethod
    def get_unmask_number(m):
        assert isinstance(m, BinaryConv2d), f'Get {m}'
        w = m.weight.detach()
        binary_w = rounding(w)
        return binary_w.sum().unsqueeze(0)
