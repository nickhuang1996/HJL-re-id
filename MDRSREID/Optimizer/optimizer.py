import torch.optim as optim


def create_optimizer(param_groups, cfg):
    """
    :param param_groups: ft_params and new_params groups
    :param cfg: cfg
    :return:
    """
    if cfg.optim.optimizer == 'sgd':
        optim_class = optim.SGD
    elif cfg.optim.optimizer == 'adam':
        optim_class = optim.Adam
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(cfg.optim.optimizer))
    if cfg.optim.optimizer == 'sgd':
        optim_kwargs = dict(weight_decay=cfg.optim.sgd.weight_decay)
        optim_kwargs['momentum'] = cfg.optim.sgd.momentum
        optim_kwargs['nesterov'] = cfg.optim.sgd.nesterov
    elif cfg.optim.optimizer == 'adam':
        optim_kwargs = {
            'weight_decay': cfg.optim.adam.weight_decay,
            'betas': (cfg.optim.adam.beta1, cfg.optim.adam.beta2),
            'eps': cfg.optim.adam.eps,
            'amsgrad': cfg.optim.adam.amsgrad
        }
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(cfg.optim.optimizer))
    return optim_class(param_groups, **optim_kwargs)
