import numpy as np


def compute_AP(index, good_index):
    '''
    :param index: np.array, 1d
    :param good_index: np.array, 1d
    :return:
    '''

    num_good = len(good_index)
    hit = np.in1d(index, good_index)
    index_hit = np.argwhere(hit == True).flatten()

    if len(index_hit) == 0:
        AP = 0
        cmc = np.zeros([len(index)])
    else:
        precision = []
        for i in range(num_good):
            precision.append(float(i + 1) / float((index_hit[i] + 1)))
        AP = np.mean(np.array(precision))
        cmc = np.zeros([len(index)])
        cmc[index_hit[0]:] = 1

    return AP, cmc
