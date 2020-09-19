from .hflip import hflip
from .resize import resize
from .pad import pad
from .random_crop import random_crop
from .to_tensor import to_tensor
from .random_erasing import random_erasing
from .random_sized_rect_crop import random_sized_rect_crop


def transforms(item, cfg, mode):
    """
    :param item: sample = deepcopy(self.items[index])
    :param cfg: cfg
    :return:

    eval() transform str to list, dict, tuple. Here is a series of the transform methods in turn.
    """
    transforms_dataset_factory = {
        'train': cfg.dataset.train,
        'test': cfg.dataset.test
    }

    if transforms_dataset_factory[mode].before_to_tensor_transform_list is not None:
        for t in transforms_dataset_factory[mode].before_to_tensor_transform_list:
            item = eval('{}(item, cfg)'.format(t))
    item = to_tensor(item, cfg)
    if transforms_dataset_factory[mode].after_to_tensor_transform_list is not None:
        for t in transforms_dataset_factory[mode].after_to_tensor_transform_list:
            item = eval('{}(item, cfg)'.format(t))
    return item
