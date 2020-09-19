from MDRSREID.utils.data_utils.transforms.numpy_transforms.normalize import normalize
import numpy as np
from sklearn import metrics as sk_metrics


def cosine_distance(x, y, cos_to_normalize=True):
    """
    Args:
        x: numpy, with shape [m, d]
        y: numpy, with shape [n, d]
        cos_to_normalize: determine if normalize 'x' and 'y' or not
    Returns:
        dist: numpy, with shape [m, n]
    """
    if cos_to_normalize:
        x = normalize(x, axis=1)
        y = normalize(y, axis=1)
    dist = - np.matmul(x, y.T)
    # Turn distance into positive value
    dist += 1
    return dist


def sklearn_cosine_distance(x, y):
    """
    :param x: query_feat
    :param y: gallery_feat
    :return:
    This is used in HOReID for feat_stage1
    """
    return 1 - sk_metrics.pairwise.cosine_distances(x, y)