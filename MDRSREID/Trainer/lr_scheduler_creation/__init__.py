from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from MDRSREID.Optimizer.WarmupLR import WarmupLR
from MDRSREID.Optimizer.WarmupMultiStepLR import WarmupMultiStepLR


def lr_scheduler_creation(cfg, optimizer, train_loader):
    if cfg.optim.phase == 'easy':
        cfg.optim.every_step = len(train_loader) * cfg.optim.every_lr_epoch
        cfg.optim.epochs = cfg.optim.easy_epochs
        lr_scheduler = StepLR(optimizer, cfg.optim.every_step, 0.1)
    elif cfg.optim.phase == 'normal':
        # cfg.optim.lr_dacay_steps is list, like [10100, 20200]
        cfg.optim.lr_decay_steps = [len(train_loader) * ep for ep in cfg.optim.lr_decay_epochs]
        cfg.optim.epochs = cfg.optim.normal_epochs
        # Use MultiStepLR to change the learning rate for multi step.
        lr_scheduler = MultiStepLR(optimizer, cfg.optim.lr_decay_steps)
    elif cfg.optim.phase == 'warmup':
        cfg.optim.warmup_steps = cfg.optim.warmup_epochs * len(train_loader)
        cfg.optim.epochs = cfg.optim.warmup_epochs
        lr_scheduler = WarmupLR(optimizer, cfg.optim.warmup_steps)
    elif cfg.optim.phase == 'warmup_multi_step':
        cfg.optim.epochs = cfg.optim.warmup_multi_step_epochs
        lr_scheduler = WarmupMultiStepLR(optimizer, cfg.optim.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)
    elif cfg.optim.phase == 'pretrain':
        cfg.optim.pretrain_new_params_steps = cfg.optim.pretrain_new_params_epochs * len(train_loader)
        cfg.optim.epochs = cfg.optim.pretrain_new_params_epochs
        lr_scheduler = None
    else:
        raise ValueError('Invalid phase {}'.format(cfg.optim.phase))
    return lr_scheduler
