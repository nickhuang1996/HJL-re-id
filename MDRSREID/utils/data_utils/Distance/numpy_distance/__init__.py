"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
from MDRSREID.utils.data_utils.Distance.numpy_distance.cosine_distance import cosine_distance
from MDRSREID.utils.data_utils.Distance.numpy_distance.cosine_distance import sklearn_cosine_distance
from MDRSREID.utils.data_utils.Distance.numpy_distance.euclidean_distance import euclidean_distance


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        dist_type: one of ['cosine', 'euclidean']
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    if dist_type == 'cosine':
        dist = cosine_distance(array1, array2, cos_to_normalize)
    elif dist_type == 'sklearn_cosine':
        dist = sklearn_cosine_distance(array1, array2)
    elif dist_type == 'euclidean':
        dist = euclidean_distance(array1, array2)
    else:
        raise NotImplementedError
    return dist


def compute_dist_with_visibility(array1, array2, vis1, vis2, dist_type='cosine', avg_by_vis_num=True):
    """Compute the euclidean or cosine distance of all pairs, considering part visibility.
    In this version, if a query image does not has some part, don't calculate distance for this part.
    If a query has one part that gallery does not have, we can optionally set the part distance to some
    prior value, e.g. the mean distance of this part.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        vis1: numpy array with shape [m1, p], p is num_parts
        vis2: numpy array with shape [m2, p], p is num_parts
        dist_type: one of ['cosine', 'euclidean']
        avg_by_vis_num: for each <query_image, gallery_image> distance, average the
            summed distance by the number of visible parts in query_image
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    err_msg = "array1.shape = {}, vis1.shape = {}, array2.shape = {}, vis2.shape = {}"\
        .format(array1.shape, vis1.shape, array2.shape, vis2.shape)
    assert array1.shape[0] == vis1.shape[0], err_msg
    assert array2.shape[0] == vis2.shape[0], err_msg
    assert vis1.shape[1] == vis2.shape[1], err_msg
    assert array1.shape[1] % vis1.shape[1] == 0, err_msg
    assert array2.shape[1] % vis2.shape[1] == 0, err_msg
    m1 = array1.shape[0]
    m2 = array2.shape[0]
    p = vis1.shape[1]
    d = int(array1.shape[1] / vis1.shape[1])

    array1 = array1.reshape([m1, p, d])
    array2 = array2.reshape([m2, p, d])
    dist = 0
    import numpy as np
    for i in range(p):
        # [m1, m2]
        dist_ = compute_dist(array1[:, i, :], array2[:, i, :], dist_type=dist_type)
        q_invisible = vis1[:, i][:, np.newaxis].repeat(m2, 1) == 0
        dist_[q_invisible] = 0
        dist += dist_
    if avg_by_vis_num:
        dist /= (np.sum(vis1, axis=1, keepdims=True) + 1e-8)
    if dist_type == 'cosine':
        # Turn distance into positive value
        dist += 1
    return dist
