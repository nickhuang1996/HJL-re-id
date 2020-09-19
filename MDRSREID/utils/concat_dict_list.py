import numpy as np
import itertools


def concat_dict_list(dict_list):
    """
    :param dict_list: 
    :return: concat_dict
    
    The values that belong to the same key can be concat together from each dict.
    Warnings:
        Each dict in dict_list should be with the same keys!
    """
    concat_dict = {}
    keys = dict_list[0].keys()
    for k in keys:
        if isinstance(dict_list[0][k], list):
            concat_dict[k] = list(itertools.chain.from_iterable([dict_[k] for dict_ in dict_list]))
        elif isinstance(dict_list[0][k], np.ndarray):
            concat_dict[k] = np.concatenate([dict_[k] for dict_ in dict_list])
        else:
            raise NotImplementedError
    return concat_dict


if __name__ == '__main__':
    import torch
    test_dict_list = []
    for i in range(3):
        test_dict_list.append({
            'feat': [i, i+1, i+2],
            'label': torch.Tensor([i]).cpu().numpy(),
            'cam': torch.Tensor([i * 2]).cpu().numpy(),
        })
        print(test_dict_list[i])
    concat_test_dict = concat_dict_list(test_dict_list)
    dic = {
        'feat':concat_test_dict['feat'],
        'label': concat_test_dict['label'],
        'cam': concat_test_dict['cam'],
    }
    print(dic)
