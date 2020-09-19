from .init_config import init_config
from .init_log import init_log
from .init_device import init_device
from MDRSREID.utils.config_utils.print_config import print_config


def pre_initialization():
    """
    :return: cfg
    """
    cfg = init_config()
    init_device(cfg)
    init_log(cfg)
    print_config(cfg, depth=0)
    return cfg
