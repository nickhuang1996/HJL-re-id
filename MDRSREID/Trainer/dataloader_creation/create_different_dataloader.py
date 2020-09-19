from MDRSREID.DataLoaders.create_dataloader import create_dataloader


def create_different_dataloader(cfg, mode=None, domain=None, name=None, authority=None, train_type=None):

    dataloader = create_dataloader(cfg,
                                   mode=mode,
                                   domain=domain,
                                   name=name,
                                   authority=authority,
                                   train_type=train_type)
    return dataloader
