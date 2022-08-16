import torch


def softmax(x1: torch.Tensor, x2: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    exp_x1 = x1.exp()
    exp_x2 = x2.exp()
    sum_x1x2 = exp_x1 + exp_x2

    return exp_x1 / sum_x1x2, exp_x2 / sum_x1x2
