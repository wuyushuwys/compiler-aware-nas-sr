"""Metrics."""

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import numpy as np


def psnr(sr, hr, shave=4):
    sr = sr.to(hr.dtype)
    sr = (sr * 255).round().clamp(0, 255) / 255
    diff = sr - hr
    if shave:
        diff = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean([-3, -2, -1])
    psnr = -10 * mse.log10()
    return psnr.mean()


def psnr_y(sr, hr, shave=4):
    sr = sr.to(hr.dtype)
    sr = (sr * 255).round().clamp(0, 255) / 255
    diff = sr - hr
    if diff.shape[1] == 3:
        filters = torch.tensor([0.257, 0.504, 0.098],
                               dtype=diff.dtype,
                               device=diff.device)
        diff = F.conv2d(diff, filters.view([1, -1, 1, 1]))
    if shave:
        diff = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean([-3, -2, -1])
    psnr = -10 * mse.log10()
    return psnr.mean()


def ssim(X, Y, shave=4):
    X = X.to(Y.dtype)
    X = (X * 255).round().clamp(0, 255) / 255
    '''
    X (groundtruth): y channel (i.e., luminance) of transformed YCbCr space of X
    Y (prediction): y channel (i.e., luminance) of transformed YCbCr space of Y
    Please follow the setting of test.py in MSRN (Multi-scale Residual Network for Image Super-Resolution ECCV2018).
    Official Link : https://github.com/MIVRC/MSRN-PyTorch
    The authors of MSRN use scikit-image's compare_ssim as the evaluation tool, 
    note that this function is quite sensitive to the argument "data_range", emprically, the larger the higher output.
    '''

    gray_coeffs = [65.738, 129.057, 25.064]
    convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    X = X.mul(convert).sum(dim=1)
    Y = Y.mul(convert).sum(dim=1)
    X = X[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)
    Y = Y[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)

    ssim = structural_similarity(X, Y,
                                 win_size=11,
                                 gaussian_weights=True,
                                 # multichannel=True,
                                 data_range=1.0,
                                 K1=0.01,
                                 K2=0.03,
                                 sigma=1.5)
    return ssim
