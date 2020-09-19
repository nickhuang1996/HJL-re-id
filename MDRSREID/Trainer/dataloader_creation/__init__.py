from MDRSREID.Trainer.dataloader_creation.create_train_dataloader import create_train_dataloader
from MDRSREID.Trainer.dataloader_creation.create_test_dataloader import create_test_dataloaders


dataloader_factory = {
    'train': create_train_dataloader,
    'test': create_test_dataloaders

}


def dataloader_creation(cfg, mode=None, domain=None, train_type=None):
    assert mode in ['train', 'test'], \
        "mode should be 'train', 'test' !!"
    return dataloader_factory[mode](cfg, mode, domain, train_type)
