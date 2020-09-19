from MDRSREID.DataLoaders.create_dataset import create_dataset
from MDRSREID.DataLoaders.Sampler import get_sampler
from torch.utils.data import DataLoader as TorchDataLoader


def create_dataloader(cfg,
                      mode=None,
                      domain=None,
                      name=None,
                      authority=None,
                      train_type=None,
                      items=None):
    """
    :param cfg:
    :param items:
    :return:

    create the dataset(search for the dataset class)
    init the sampler
    create the loader
    """
    train_name_factory = {
        'source': cfg.dataset.train.source,
        'target': cfg.dataset.train.target,
    }
    if mode is 'train':
        name = train_name_factory[domain].name
    dataset = create_dataset(cfg,
                             mode=mode,
                             domain=domain,
                             name=name,
                             authority=authority,
                             train_type=train_type,
                             items=items)
    # from DataLoaders.Datasets.market1501 import Market1501
    # dataset = Market1501(cfg, items=items)
    sampler_factory = {
        'train': cfg.dataloader.train,
        'test': cfg.dataloader.test
    }
    sampler = get_sampler(cfg, dataset, sampler_factory[mode])

    data_loader = TorchDataLoader(
        dataset=dataset,
        batch_size=sampler_factory[mode].batch_size,
        sampler=sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        drop_last=sampler_factory[mode].drop_last,
    )
    return data_loader
