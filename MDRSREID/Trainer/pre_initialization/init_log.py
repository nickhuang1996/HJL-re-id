from MDRSREID.utils.log_utils.log import time_str
from MDRSREID.utils.log_utils.log import ReDirectSTD
import os.path as osp


def init_log(cfg):
    log_cfg = cfg.log
    # Redirect logs to both console and file.
    time_string = time_str()
    ReDirectSTD(osp.join(log_cfg.exp_dir, 'stdout_{}.txt'.format(time_string)), 'stdout', True)
    ReDirectSTD(osp.join(log_cfg.exp_dir, 'stderr_{}.txt'.format(time_string)), 'stderr', True)
    print('=> Experiment Output Directory: {}'.format(log_cfg.exp_dir))
    import torch
    print('[PYTORCH VERSION]:', torch.__version__)
    log_cfg.ckpt_file = osp.join(log_cfg.exp_dir, 'ckpt.pth')
    log_cfg.score_file = osp.join(log_cfg.exp_dir, 'score_{}.txt'.format(time_string))


if __name__ == '__main__':
    from MDRSREID.Trainer.pre_initialization.init_config import init_config
    cfg = init_config()
    init_log(cfg)
    print(cfg.log)
