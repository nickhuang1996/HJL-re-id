import numpy as np
import torchvision.transforms.functional as F


def hflip(item, cfg):
    # Tricky!! random.random() can not reproduce the score of np.random.random(),
    # dropping ~1% for both Market1501 and Duke GlobalPool.
    # if random.random() < 0.5:
    if np.random.random() < 0.5:
        item['im'] = F.hflip(item['im'])
        if 'pose_landmark_mask' in item:
            item['pose_landmark_mask'] = np.flip(item['pose_landmark_mask'], 2)
        if 'ps_label' in item:
            item['ps_label'] = F.hflip(item['ps_label'])
        # if 'ps_label_list' in item:
        #     item['ps_label_list'] = [F.hflip(item['ps_label_list'][i] for i in range(len(item['ps_label_list'])))]
    return item
