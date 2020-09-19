"""I import the Sampler here."""

from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from .RamdomIdentitySampler import RandomIdentitySampler
from .RamdomIdentitySampler2 import RandomIdentitySampler2


def get_sampler(cfg, dataset, sampler_dict):
    """
    :param cfg: cfg.batch_type
    :return:

    According to the batch_type to choose a sampler.
    """

    if sampler_dict.batch_type == 'seq':
        sampler = SequentialSampler(dataset)
    elif sampler_dict.batch_type == 'random':
        sampler = RandomSampler(dataset)
    elif sampler_dict.batch_type == 'pk':
        # this is the new sampler, each person id choices cfg.pk.k images
        sampler = RandomIdentitySampler(dataset, cfg.dataloader.pk.k)
    elif sampler_dict.batch_type == 'pk2':
        # this is the new sampler, each person id choices cfg.pk.k images
        sampler = RandomIdentitySampler2(dataset, sampler_dict.batch_id, sampler_dict.batch_image)
    else:
        raise NotImplementedError
    return sampler
