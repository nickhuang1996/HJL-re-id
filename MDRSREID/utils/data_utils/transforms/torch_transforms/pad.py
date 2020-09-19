import torchvision.transforms.functional as F


def pad(item, cfg):
    """Pad the given PIL Image on all sides with speficified padding mode and fill value"""
    item['im'] = F.pad(item['im'], cfg.dataset.im.pad, fill=0, padding_mode='constant')
    if 'ps_label' in item:
        item['ps_label'] = F.pad(item['im'], cfg.dataset.ps_label.pad, fill=0, padding_mode='constant')
    return item
