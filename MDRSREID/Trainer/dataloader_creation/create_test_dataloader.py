from MDRSREID.Trainer.dataloader_creation.create_different_dataloader import create_different_dataloader
from collections import OrderedDict


def create_test_dataloaders(cfg, mode, domain=None, train_type=None):
    """
    :param cfg:
    :param mode:
    :param domain: it is just for train source or target.
    :param train_type: it is just for train type: Supervised or Unsupervised
    :return:
    """
    test_dataloader = OrderedDict()
    for index, name in enumerate(cfg.dataset.test.names):
        query_authority = cfg.dataset.test.query_authorities[index] if hasattr(cfg.dataset.test, 'query_authorities') else 'query'
        test_dataloader[name] = {
            'query': create_different_dataloader(cfg,
                                                 mode=mode,
                                                 domain=domain,
                                                 name=name,
                                                 authority=query_authority,
                                                 train_type=train_type),
            'gallery': create_different_dataloader(cfg,
                                                   mode=mode,
                                                   domain=domain,
                                                   name=name,
                                                   authority='gallery',
                                                   train_type=train_type)
        }
    return test_dataloader

