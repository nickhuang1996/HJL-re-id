import math
import random
from PIL import Image
import numpy as np
import cv2


class RectScale(object):
    def __init__(self,
                 height,
                 width,
                 interpolation=cv2.INTER_LINEAR  # Image.BILINEAR
                 ):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        # return img.resize((self.width, self.height), self.interpolation)
        return Image.fromarray(
            cv2.resize(np.array(img), (self.width, self.height), interpolation=self.interpolation))


def random_sized_rect_crop(item, cfg):
    """
    :param item:
    :param cfg:
    :return:
    Return random sized rectangle crop item['im'] and item['ps_label']
    """
    (im_height, im_width) = cfg.dataset.im.h_w
    (ps_label_height, ps_label_width) = cfg.dataset.ps_label.h_w

    for attempt in range(10):
        target_area_ratio = random.uniform(0.64, 1.0)
        aspect_ratio = random.uniform(2, 3)
        item['im'], crop_ratio_dict = attempt_body(item['im'],
                                                   target_area_ratio=target_area_ratio,
                                                   aspect_ratio=aspect_ratio,
                                                   height=im_height,
                                                   width=im_width
                                                   )
        if crop_ratio_dict is not None:
            if 'ps_label' in item:
                item['ps_label'], _ = attempt_body(item['ps_label'],
                                                   height=ps_label_height,
                                                   width=ps_label_width,
                                                   crop_ratio_dict=crop_ratio_dict
                                                   )
            return item

    # Fallback
    scale_im = RectScale(im_height, im_width,
                         interpolation=cv2.INTER_LINEAR)  # Image.BILINEAR
    scale_ps_label = RectScale(ps_label_height, ps_label_width,
                               interpolation=cv2.INTER_LINEAR)
    item['im'] = scale_im(item['im'])
    item['ps_label'] = scale_ps_label(item['ps_label'])
    return item


def attempt_body(img,
                 target_area_ratio=None,
                 aspect_ratio=None,
                 height=None,
                 width=None,
                 crop_ratio_dict=None):
    """
    :param img: item['im'] or item['ps_label']
    :param target_area_ratio: the area size of target
    :param aspect_ratio: for the height and width of the target area
    :param height:
    :param width:
    :param crop_ratio_dict: Only record the item['im'] crop ratios
    :return: cropped img area

    Attention:
        The 'ps_label' images should be cropped with the same ratio of the 'im' images.
    """
    if crop_ratio_dict is None:
        area = img.size[0] * img.size[1]
        target_area = target_area_ratio * area

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
    else:
        crop_h_ratio = crop_ratio_dict['crop_h_ratio']
        crop_w_ratio = crop_ratio_dict['crop_w_ratio']

        h = int(round(img.size[1] * crop_h_ratio))
        w = int(round(img.size[0] * crop_w_ratio))

    if w <= img.size[0] and h <= img.size[1]:
        if crop_ratio_dict is None:
            x1 = random.randint(0, img.size[0] - w)
            y1 = random.randint(0, img.size[1] - h)
            crop_ratio_dict = {
                'crop_w_ratio': w / img.size[0],
                'crop_h_ratio': h / img.size[1],
                'crop_x1_ratio': x1 / img.size[0],
                'crop_y1_ratio': y1 / img.size[1],
            }
        else:
            crop_x1_ratio = crop_ratio_dict['crop_x1_ratio']
            crop_y1_ratio = crop_ratio_dict['crop_y1_ratio']
            x1 = int(round(img.size[0] * crop_x1_ratio))
            y1 = int(round(img.size[1] * crop_y1_ratio))

        img = img.crop((x1, y1, x1 + w, y1 + h))
        assert (img.size == (w, h))

        img = Image.fromarray(
            cv2.resize(np.array(img), (width, height), interpolation=cv2.INTER_LINEAR))

    return img, crop_ratio_dict
