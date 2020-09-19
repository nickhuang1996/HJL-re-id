import random
import math


def random_erasing(item, cfg):
    """
    :param item:
    :param cfg:
    :return:
    Random erasing the item['im'] and item['ps_label']
    """
    im_finish = 0
    ps_label_finish = 0

    mean = cfg.dataset.im.mean

    if random.uniform(0, 1) > cfg.dataset.im.random_erasing.epsilon:
        return item

    for attempt in range(100):
        if cfg.dataset.im.random_erasing.proportion is True:
            target_area_ratio = random.uniform(0.02, 0.2)
            aspect_ratio = random.uniform(0.3, 3)
            item['im'], crop_ratio_dict = attempt_body(item['im'],
                                                       mean,
                                                       target_area_ratio=target_area_ratio,
                                                       aspect_ratio=aspect_ratio)
            if crop_ratio_dict is not None:
                if 'ps_label' in item:
                    item['ps_label'], _ = attempt_body(item['ps_label'],
                                                       mean,
                                                       crop_ratio_dict=crop_ratio_dict)
                return item
        else:
            if im_finish == 0:
                target_area_ratio = random.uniform(0.02, 0.2)
                aspect_ratio = random.uniform(0.3, 3)
                item['im'], crop_ratio_dict = attempt_body(item['im'],
                                                           mean,
                                                           target_area_ratio=target_area_ratio,
                                                           aspect_ratio=aspect_ratio)
                if crop_ratio_dict is not None:
                    im_finish = 1
            if ps_label_finish == 0:
                target_area_ratio = random.uniform(0.02, 0.2)
                aspect_ratio = random.uniform(0.3, 3)
                item['ps_label'], crop_ratio_dict = attempt_body(item['ps_label'],
                                                                 mean,
                                                                 target_area_ratio=target_area_ratio,
                                                                 aspect_ratio=aspect_ratio)
                if crop_ratio_dict is not None:
                    ps_label_finish = 1
            all_finish = im_finish & ps_label_finish
            if all_finish:
                return item
    return item


def attempt_body(img,
                 mean,
                 target_area_ratio=None,
                 aspect_ratio=None,
                 crop_ratio_dict=None
                 ):
    """
    :param img: item['im'] or item['ps_label']
    :param mean: for the uniform
    :param target_area_ratio:
    :param aspect_ratio:
    :param crop_ratio_dict: Only record the item['im'] crop ratios
    :return:

    Attention:
        1.The 'ps_label' images is not like 'im' images [3, height, width], but [height, width].
        Thus, I unsqueeze the pytorch tensor to [1, height, width] and after random erasing process is finished, I
        squeeze the tensor like original one.
        2.The 'ps_label' images should be cropped with the same ratio of the 'im' images.
    """
    if crop_ratio_dict is None:
        if len(list(img.size())) == 2:
            img = img.unsqueeze(dim=0).contiguous()
        else:
            assert img.size()[0] == 3, "{}".format(img.size())
        area = img.size()[1] * img.size()[2]
        target_area = target_area_ratio * area

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
    else:
        crop_h_ratio = crop_ratio_dict['crop_h_ratio']
        crop_w_ratio = crop_ratio_dict['crop_w_ratio']
        if len(list(img.size())) == 2:
            img = img.unsqueeze(dim=0).contiguous()
        h = int(round(img.size()[1] * crop_h_ratio))
        w = int(round(img.size()[2] * crop_w_ratio))

    if w <= img.size()[2] and h <= img.size()[1]:
        if crop_ratio_dict is None:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            crop_ratio_dict = {
                'crop_w_ratio': w / img.size()[2],
                'crop_h_ratio': h / img.size()[1],
                'crop_x1_ratio': x1 / img.size()[1],
                'crop_y1_ratio': y1 / img.size()[2],
            }
        else:
            crop_x1_ratio = crop_ratio_dict['crop_x1_ratio']
            crop_y1_ratio = crop_ratio_dict['crop_y1_ratio']
            x1 = int(round(img.size()[1] * crop_x1_ratio))
            y1 = int(round(img.size()[2] * crop_y1_ratio))
        if img.size()[0] == 3:
            img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            img[1, x1:x1 + h, y1:y1 + w] = mean[1]
            img[2, x1:x1 + h, y1:y1 + w] = mean[2]
        else:
            img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            img = img.squeeze(dim=0).contiguous()
    else:
        if img.size()[0] == 1:
            img = img.squeeze(dim=0).contiguous()
    return img, crop_ratio_dict
