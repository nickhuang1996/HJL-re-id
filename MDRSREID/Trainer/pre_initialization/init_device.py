from MDRSREID.utils.device_utils.get_default_device import get_default_device


def init_device(cfg):
    """
    :param cfg:
    :return: cfg.eval.device
    """
    cfg.device = get_default_device()
    cfg.eval.device = cfg.device


if __name__ == '__main__':
    from MDRSREID.Trainer.pre_initialization.init_config import init_config
    cfg = init_config()
    init_device(cfg)
    assert cfg.eval.device == cfg.device
    print('device init successfully.')


