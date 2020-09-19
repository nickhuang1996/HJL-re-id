import numpy as np
from .compute_AP import compute_AP
from .in1d import in1d
from .notin1d import notin1d


def evaluate(index, query_cam, query_label, gallery_cam, gallery_label, dist):
    junk_index_1 = in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
    junk_index_2 = np.argwhere(gallery_label == -1)
    junk_index = np.append(junk_index_1, junk_index_2)

    good_index = in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
    index_wo_junk = notin1d(index, junk_index)

    return compute_AP(index_wo_junk, good_index)
