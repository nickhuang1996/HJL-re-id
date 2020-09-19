from MDRSREID.Trainer.dataloader_creation.create_different_dataloader import create_different_dataloader


def create_train_dataloader(cfg, mode, domain, train_type):
    """
    :param cfg:
    :return: dataloader
    """
    train_dataloader = create_different_dataloader(cfg, mode, domain, authority='train', train_type=train_type)
    cfg.model.num_classes = train_dataloader.dataset.num_ids  # 751
    return train_dataloader
