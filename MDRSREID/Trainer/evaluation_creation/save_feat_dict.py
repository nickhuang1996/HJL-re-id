import scipy.io
import os.path as osp
import warnings


def save_feat_dict(is_reproduction, feat_dict, feat_dict_path):
    if not osp.exists(feat_dict_path):
        scipy.io.savemat(feat_dict_path, feat_dict)
        print('Saving feat dict into {} has been finished.'.format(feat_dict_path))
    elif osp.exists(feat_dict_path):
        print("{} has been existed.".format(feat_dict_path))
        if is_reproduction:
            warnings.warn("`is_reproduction` is better to set `False` if {} has been existed!!".format(feat_dict_path))
    if is_reproduction:
        scipy.io.savemat(feat_dict_path, feat_dict)
        print('Overwriting feat dict into {} has been finished.'.format(feat_dict_path))
    return
