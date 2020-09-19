import torch
import numpy as np
import torchvision.transforms.functional as F


def to_tensor(item, cfg):
    """
    :param item: sample = deepcopy(self.samples[index])
    :param cfg: cfg
    :return:

    There are always 2 transform ways:
        F.to_tensor
        F.normalize
    """
    item['im'] = F.to_tensor(item['im'])
    item['im'] = F.normalize(item['im'], cfg.dataset.im.mean, cfg.dataset.im.std)
    if 'pose_landmark_mask' in item:
        item['pose_landmark_mask'] = torch.from_numpy(item['pose_landmark_mask'].copy()).float()
    if 'ps_label' in item:
        item['ps_label'] = torch.from_numpy(np.array(item['ps_label'])).long()
    return item
