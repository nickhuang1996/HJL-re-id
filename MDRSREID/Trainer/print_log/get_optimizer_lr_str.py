

def get_optimizer_lr_str(cfg, optimizer):
    if cfg.log.only_base_lr is False:
        lr_strs = ['{:.6f}'.format(g['lr']).rstrip('0') for g in optimizer.param_groups]
        lr_str = ', '.join(lr_strs)
    else:
        lr_str = '{:.6f}'.format(optimizer.param_groups[0]['lr'])
    return lr_str
