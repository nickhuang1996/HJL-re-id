import time
from MDRSREID.utils.log_utils.log import join_str
from .get_optimizer_lr_str import get_optimizer_lr_str


def print_log(cfg, current_ep, current_step, optimizer, loss_functions, analyze_functions, epoch_start_time):
    time_log = 'Ep {}, Step {}, {:.2f}s'.format(current_ep + 1, current_step + 1,
                                                time.time() - epoch_start_time)
    lr_log = 'lr {}'.format(get_optimizer_lr_str(cfg, optimizer))
    loss_meter_log = join_str([m.avg_str for lf in loss_functions.values() for m in lf.meter_dict.values()], ', ')
    if analyze_functions is not None:
        analyze_meter_log = join_str([m.avg_str for lf in analyze_functions.values() for m in lf.meter_dict.values()], ', ')
    else:
        analyze_meter_log = None
    if analyze_meter_log is not None:
        log = join_str([time_log, lr_log, loss_meter_log, analyze_meter_log], ', ')
    else:
        log = join_str([time_log, lr_log, loss_meter_log], ', ')
    return print(log)
