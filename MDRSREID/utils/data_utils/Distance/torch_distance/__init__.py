from MDRSREID.utils.data_utils.Distance.torch_distance.cosine_distance import cosine_distance
from MDRSREID.utils.data_utils.Distance.torch_distance.euclidean_distance import euclidean_distance


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True, opposite=False):
    """
    Args:
        array1: pytorch tensor, with shape [m, d]
        array2: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    if dist_type == 'cosine':
        dist = cosine_distance(array1, array2, cos_to_normalize, opposite)
    elif dist_type == 'euclidean':
        dist = euclidean_distance(array1, array2)
    else:
        raise NotImplementedError
    return dist
