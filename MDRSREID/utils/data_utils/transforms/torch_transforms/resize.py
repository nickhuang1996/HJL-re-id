import cv2
import numpy as np
from PIL import Image


def resize_3d_np_array(maps, resize_h_w, interpolation):
    """maps: np array with shape [C, H, W], dtype is not restricted"""
    return np.stack([cv2.resize(m, tuple(resize_h_w[::-1]), interpolation=interpolation) for m in maps])


def resize(item, cfg):
    """
    :param item: sample = deepcopy(self.samples[index])
    :param cfg: cfg
    :return:

    Resize the image by cv2.resize, but we use the Image.fromarray to get the RGB image.
    """
    item['im'] = Image.fromarray(
        cv2.resize(np.array(item['im']), tuple(cfg.dataset.im.h_w[::-1]), interpolation=cv2.INTER_LINEAR))
    # if 'pose_landmark_mask' in item:
    #     item['pose_landmark_mask'] = resize_3d_np_array(item['pose_landmark_mask'], cfg.dataset.pose_landmark_mask.h_w, cv2.INTER_NEAREST)
    if 'ps_label' in item:
        item['ps_label'] = Image.fromarray(
            cv2.resize(np.array(item['ps_label']), tuple(cfg.dataset.ps_label.h_w[::-1]), cv2.INTER_NEAREST), mode='L')
    # if 'ps_label_list' in item:
    #     item['ps_label_list'] = [Image.fromarray(
    #         cv2.resize(np.array(item['ps_label_list'][i]), tuple(cfg.ps_label.h_w[::-1]), cv2.INTER_NEAREST), mode='L') for i in range(len(item['ps_label_list']))]
    return item
