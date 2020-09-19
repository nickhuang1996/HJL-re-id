import numpy as np


def euclidean_distance(x, y):
    """
    Args:
        x: numpy, with shape [m, d]
        y: numpy, with shape [n, d]
    Returns:
        dist: numpy, with shape [m, n]
    """
    # shape [m1, 1]
    square1 = np.sum(np.square(x), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(y), axis=1)[np.newaxis, ...]
    dist = - 2 * np.matmul(x, y.T) + square1 + square2
    dist[dist < 0] = 0
    # Print('Debug why there is warning in np.sqrt')
    # np.seterr(all='raise')
    # for x in dist.flatten():
    #     try:
    #         np.sqrt(x)
    #     except:
    #         print(x)
    # Setting `out=dist` saves 1x memory size of `dist`
    np.sqrt(dist, out=dist)
    return dist
