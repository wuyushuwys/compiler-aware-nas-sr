"""Meters."""
import time
import datetime


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.end_time = self.start_time
        self.sum = 0
        self.avg = 0
        self.count = 0
        self.remain_time = 0

    def update(self, n=1):
        self.end_time = time.time()
        self.sum = self.end_time - self.start_time
        self.count += n
        self.avg = self.sum / self.count

    def update_count(self, count):
        self.end_time = time.time()
        self.sum = self.end_time - self.start_time
        self.count = count
        self.avg = self.sum / self.count

    def complete_time(self, remain_batch):
        self.remain_time = datetime.timedelta(seconds=int(self.avg * remain_batch))
