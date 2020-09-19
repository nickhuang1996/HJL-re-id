from MDRSREID.Trainer.pre_initialization.init_config import init_config
from easydict import EasyDict


def print_config(cfg, depth=0):
    """
    :param cfg:
    :param depth: print several tabs to establish the config
    :return:
    """
    tab_str = '\t'
    for key, item in cfg.items():
        if type(item) == EasyDict:
            print(tab_str * depth, key, ':')
            depth += 1
            print_config(item, depth)
            depth -= 1
        else:
            print(tab_str * depth, key, ':', item)


if __name__ == '__main__':
    cfg = init_config()
    print_config(cfg, depth=0)
