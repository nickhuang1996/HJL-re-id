import random

def random_crop(item, cfg):
    """
    Args:
        img (PIL Image): Image to be cropped.

    Returns:
        PIL Image: Cropped image.
    """

    w, h = item['im'].size
    output_size = cfg.dataset.im.random_crop.output_size
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    else:
        output_size = output_size
    th, tw = output_size
    if w == tw and h == th:
        i = 0
        j = 0
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    item['im'] = item['im'].crop((j, i, j + tw, i + th))

    return item
