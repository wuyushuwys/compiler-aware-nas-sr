import torch
from .speed_estimator import BlockBSpeedEstimator


def get_ori_speed(num_blocks=4, num_residual_units=12):
    with torch.no_grad():
        speed_estimator = BlockBSpeedEstimator('tensor').cpu()
        expand = 6
        linear = 0.84
        channels_number = torch.tensor([num_residual_units, num_residual_units * expand,
                                        int(num_residual_units * linear), num_residual_units], dtype=torch.float).cpu()
        block_speed = speed_estimator.estimateByChannelNum(channels_number)
        total_speed = num_blocks * block_speed
    return total_speed.item()
