import warnings
import argparse
import importlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image

import common.meters
import common.modes
import common.metrics

from utils.estimate import test
from utils import logging_tool
from collections import OrderedDict

import models
from utils import attr_extractor, loss_printer
from loss_config import update_weight
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

warnings.simplefilter("ignore", UserWarning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model,
          optimizer,
          train_data_loader,
          criterions,
          writer,
          params,
          epoch,
          device):
    """
    Train model
    """

    loss_meter = common.meters.AverageMeter()
    time_meter = common.meters.TimeMeter()
    block_meter = common.meters.AverageMeter()
    model.train()

    nb = len(train_data_loader)
    losses = {}

    for batch_idx, (lr, hr) in enumerate(train_data_loader):

        hr_label = hr.clone().detach().to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)

        total_batches = (epoch - 1) * nb + batch_idx

        # Train
        optimizer.zero_grad()
        sr = model(lr)
        loss = 0

        loss_sr_l1 = params.weight_sr_l1 * criterions['l1'](sr, hr_label)
        losses['l1'] = loss_sr_l1
        loss += loss_sr_l1

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        time_meter.update()
        loss_meter.update(loss.item(), sr.size(0))

        if (batch_idx % params.log_steps == 0) and device == 0:
            time_meter.complete_time(nb - batch_idx)

            writer.add_scalar('training_loss', loss.item(), total_batches)
            writer.add_scalar('Loss/l1', loss_sr_l1, total_batches)
            total_epochs = params.epochs
            s = f"## Epoch: {epoch:{' '}{'>'}{2}d}/{total_epochs:d}\t" \
                f"Iters:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb * 100:.2f}%)\t" \
                f"Epoch-est. {time_meter.remain_time}\t" \
                f"Loss: {loss.item():.6f}\t" \
                f"{loss_printer(losses)}"
            print(s)

    if device == 0:
        writer.add_scalar('training_loss_smooth', loss_meter.avg, epoch)
        logging.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{params.epochs} finished.\tLoss: {loss_meter.avg:.6f}")

    # Save image samples per epoch
    if device == 0:
        save_image(sr, os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_output.png'))  # save output
        save_image(hr_label, os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_target.png'))  # save target


def evaluation(model, eval_data_loaders, epoch, writer, device):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param writer: tb_writer
    :param device: index of device
    """
    if writer and device == 0:
        for eval_data_name, eval_data_loader in eval_data_loaders:
            handle_dict = {'job_dir': writer.get_logdir(),
                           'epoch': epoch,
                           'eval_data_name': eval_data_name, }
            psnr, psnr_y, ssim, speed = test(eval_data_loader, model, gpu=device)
            writer.add_scalar(f"{eval_data_name}/PSNR", psnr, epoch)
            writer.add_scalar(f"{eval_data_name}/PSNR_Y", psnr_y, epoch)
            writer.add_scalar(f"{eval_data_name}/SSIM", ssim, epoch)
            logging.info(f"##\tEval: {eval_data_name}\t"
                         f"PSNR: {psnr:.4f}\tPSNR_Y: {psnr_y:.4f}\tSSIM: {ssim:.4f}")
        try:
            writer.add_scalar(f"Arch/Speed", speed.item(), epoch)
            logging.info(f"##\tModel Speed {speed.item():.04f}", device=device)
        except AttributeError:
            pass
        logging.info(f"Finish Epoch {epoch} Evaluation\n", device=device)


def trainer_preparation(model, learning_rate, epochs):
    # All optimizer functions and scheduler functions
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(epochs * 1 / 2),
                                                           int(epochs * 9 / 10)],
                                               gamma=0.3)
    return optimizer, scheduler


def main(params, logging):
    logging.info(f"Using GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")
    dataset_module = importlib.import_module(f'datasets.{params.dataset}' if params.dataset else 'datasets')

    # Enable cudnn Optimization for static network structure
    torch.backends.cudnn.benchmark = True

    if params.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_seed(local_rank)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        params.world_size = torch.distributed.get_world_size()

    # Set device
    device = local_rank if params.distributed else 0
    update_weight(params=params)
    logging.info(attr_extractor(params), device=device)

    # Create job and tb_writer
    writer = SummaryWriter(params.job_dir) if device == 0 else None

    # Load train datasetcd
    train_dataset = dataset_module.get_dataset(common.modes.TRAIN, params)

    # Load eval dataset
    if params.eval_datasets:
        eval_datasets = []
        for eval_dataset in params.eval_datasets:
            eval_dataset_module = importlib.import_module(f'datasets.{eval_dataset}')
            eval_datasets.append((eval_dataset, eval_dataset_module.get_dataset(common.modes.EVAL, params)))
    else:
        eval_datasets = [(params.dataset, dataset_module.get_dataset(common.modes.EVAL, params))]

    # Dataloader sampler set to None for single GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if params.distributed else None
    eval_sampler = None

    if params.debug:
        logging.info('Enable anomaly detect', device=device)
        logging.warning('Debug mode is super slow, set epochs to 1', device=device)
        params.epochs = 1
        torch.autograd.set_detect_anomaly(params.debug)

    # Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   num_workers=params.num_data_threads,
                                   batch_size=params.train_batch_size,
                                   shuffle=(train_sampler is None),
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=train_sampler)

    eval_kwargs = {"num_workers": params.num_data_threads,
                   "batch_size": params.eval_batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'pin_memory': True,
                   'sampler': eval_sampler}
    eval_data_loaders = [(data_name, DataLoader(dataset=dataset, **eval_kwargs)) for data_name, dataset in
                         eval_datasets]

    # Create Model

    model = models.get_model(params=params)
    logging.info(f"\n{model}", is_print=False, device=device)

    if params.distributed:
        logging.info("Distributed Training", device=device)
        params.learning_rate *= params.world_size

    # Loss function
    criterions = OrderedDict()
    criterions['l1'] = nn.L1Loss().to(device)

    if params.eval_model:
        model.load_state_dict(torch.load(params.eval_model, map_location='cpu'))

    if device == 0:
        for folder in ['results', 'weights', 'ckpt', 'eval']:
            folder_path = os.path.join(params.job_dir, folder)
            if os.path.exists(folder_path):
                logging.warning(f'{os.path.join(params.job_dir, folder)} already exists', device=device)
            else:
                os.mkdir(folder_path)
                logging.info(f'Create {folder_path}', device=device)

    # Train
    epoch = 0
    end_epoch = 0

    if params.distributed:
        model = DDP(model.to(device), device_ids=[device], output_device=device)
    else:
        model.to(device)

    if params.eval_model:
        evaluation(model, eval_data_loaders, 0, writer, device=device)
        return

    # All optimizer functions and scheduler functions
    optimizer, scheduler = trainer_preparation(model=model,
                                               learning_rate=params.learning_rate,
                                               epochs=params.epochs)
    end_epoch += params.epochs
    for epoch in range(epoch + 1, end_epoch + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train(model=model, optimizer=optimizer,
              train_data_loader=train_data_loader,
              criterions=criterions,
              writer=writer, params=params, epoch=epoch, device=device)
        evaluation(model, eval_data_loaders, epoch, writer, device=device)
        if device == 0:
            torch.save(model.state_dict() if not params.distributed else model.module.state_dict(),
                       os.path.join(params.job_dir, 'weights', 'models.pt'))
            torch.save({"epoch": epoch,
                        "state_dict": model.state_dict() if not params.distributed else model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        }, os.path.join(params.job_dir, "ckpt", f"latest.pth"))
        scheduler.step()

    if not params.eval_only and device == 0:
        writer.close()

    logging.info(f"Finish Training", device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='NAS_MODEL', type=str, required=True,
                        help='model type: NAS_MODEL / BASIC_MODEL')
    parser.add_argument('--dataset', default=None, type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('--job_dir', default=None, type=str, required=True,
                        help='Directory to write checkpoints and export models.')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='File path to load checkpoint.')
    parser.add_argument('--model_weight', default=None, type=str,
                        help='path to weight')
    parser.add_argument('--scheduler_type', default='multi_step', type=str,
                        help="LR scheduler type: cosine, multi_step")

    # evaluation
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Running evaluation only.')
    parser.add_argument('--eval_model', default=None, type=str,
                        help='Path to evaluation model.')
    parser.add_argument('--eval_datasets', default=None, type=str, nargs='+',
                        help='Dataset names for evaluation.')

    # Experiment arguments
    parser.add_argument('--speed_target', default=40, type=float,
                        help='speed target in ms.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of search epochs.')
    parser.add_argument('--width_epochs', default=0, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--finetune_epochs', default=30, type=int,
                        help='Number of fine-tune epochs.')
    parser.add_argument('--log_steps', default=100, type=int,
                        help='Number of steps for training logging.')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='Recognized opt_levels are O0, O1, O2, and O3')

    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume training from ckpt')
    parser.add_argument('--warmup_lr', default=False, action='store_true',
                        help='Using warmup lr')

    # Verbose
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Increasing output verbosity.', )
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug mode. Enable torch anomaly check')

    # for multi GPU
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='Distributed training')

    # Parse arguments
    args, _ = parser.parse_known_args()
    logging = logging_tool.LoggingTool(file_path=args.job_dir, verbose=args.verbose)

    # Update args
    importlib.import_module(f'datasets.{args.dataset}' if args.dataset else 'datasets').update_argparser(parser)
    models.update_argparser(parser)

    # parsing args
    params = parser.parse_args()

    main(params, logging)
