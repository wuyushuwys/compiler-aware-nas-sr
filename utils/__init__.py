def attr_extractor(obj):
    attrs = list(filter(lambda x: not x.startswith('_'), dir(obj)))  # Remove default and help attributes
    attr_dict = dict()
    info_len = 30
    string = f"\n{'INFO':{'*'}{'^'}{80}s}\n"
    str_head = '** '
    for name in attrs:
        attr_dict[name] = getattr(obj, name)
    for k, v in attr_dict.items():
        v_str = str(v)
        string += f"{str_head}{f'{k}:':{''}{'<'}{info_len}s}{v_str}\n"

    string += f"{'':{'*'}{'^'}{80}s}\n"
    return string


def loss_printer(loss_dict: dict):
    s = ''
    for k, v in loss_dict.items():
        if k != 'loss':
            s += f"{k}:{v.item():.4e}  " if hasattr(v, 'item') else f"{k}:{v:.4e}  "
    return f"[{s.rstrip()}]"


class SpeedScheduler:
    # Gradually change the latency target
    def __init__(self, search_epoch, total_speed, target_speed, gamma=0.75):
        self.search_epoch = search_epoch
        self.total_speed = total_speed
        self.target_speed = target_speed
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        # print(f"{self.epoch}<->{self.search_epoch * self.gamma}")
        if self.epoch > self.search_epoch * self.gamma:
            return self.target_speed
        else:
            return self.target_speed + (self.search_epoch * self.gamma - self.epoch) * \
                (self.total_speed - self.target_speed) / (self.search_epoch * self.gamma)
