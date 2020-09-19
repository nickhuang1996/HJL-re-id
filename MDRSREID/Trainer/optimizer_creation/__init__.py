from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device
from MDRSREID.Optimizer.optimizer import create_optimizer


def optimizer_creation(cfg, model):
    """
    :return:

    The parameters has been split to 2 groups:
        ft_params: always ResNet parameters.
        new_params: always many layers parameters after ResNet.
    ft_params use cfg.ft_lr
    new_params use cfg.new_params_lr

    They are pairs by OrderedDict to save.
    """
    param_groups = model.get_param_groups()
    optimizer = create_optimizer(param_groups, cfg)
    recursive_to_device(optimizer.state_dict(), cfg.device)
    return optimizer
