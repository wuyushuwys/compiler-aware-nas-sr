import os
import torch
import torch.nn as nn
import torch.utils.data

from common.metrics import psnr, psnr_y, ssim
from torchvision.utils import save_image


def test(dataloader: torch.utils.data.DataLoader, model: nn.Module, gpu: torch.device, f: dict):
    # switch eval mode.
    model.eval()
    total_psnr_value = 0.
    total_psnr_y_value = 0.
    total_ssim_value = 0.
    total = len(dataloader)
    with torch.no_grad():
        for i, (name, lr, hr) in enumerate(dataloader):

            if torch.cuda.is_available():
                hr = hr.to(gpu, non_blocking=True)
                lr = lr.to(gpu, non_blocking=True)
            if hr.shape[1] == 1:
                # gray scale to 3 channel
                hr = torch.stack([hr.squeeze(0), hr.squeeze(0), hr.squeeze(0)], 1)
                lr = torch.stack([lr.squeeze(0), lr.squeeze(0), lr.squeeze(0)], 1)

            output = model(lr)
            if len(output) == 2:
                sr, speed_accu = output
            else:
                sr, speed_accu = output, None
            path = f"{f['job_dir']}/eval/{f['eval_data_name']}"
            os.makedirs(path, exist_ok=True)
            save_image(sr.clamp(0, 1), f"{path}/{name[0]}.png")
            scale = model.scale if not hasattr(model, 'module') else model.module.scale
            # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
            total_psnr_y_value += psnr_y(sr, hr, shave=scale)
            total_psnr_value += psnr(sr, hr, shave=scale + 6)
            # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
            total_ssim_value += ssim(sr, hr, shave=scale)
            # Shave boarder
    out = total_psnr_value / total, total_psnr_y_value / total, total_ssim_value / total, speed_accu
    return out
