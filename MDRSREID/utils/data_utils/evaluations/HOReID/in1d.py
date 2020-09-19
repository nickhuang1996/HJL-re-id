import numpy as np


def in1d(array1, array2, invert=False):
    '''
    :param set1: np.array, 1d
    :param set2: np.array, 1d
    :return:
    '''
    mask = np.in1d(array1, array2, invert=invert)
    return array1[mask]
