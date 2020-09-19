import scipy.io
import os.path as osp
import warnings


def load_feat_dict(feat_dict_path):
    if osp.exists(feat_dict_path):
        print('Loading feat dict mat from {}...'.format(feat_dict_path))
        feat_dict = scipy.io.loadmat(feat_dict_path)
        is_reproduction, feat_dict = _transform_feat_dict(feat_dict)
        return is_reproduction, feat_dict
    else:
        warnings.warn("Feat dict mat {} is not existed!!".format(feat_dict_path))
        return None, None


def _transform_feat_dict(feat_dict):
    feat_dict.pop('__header__')
    feat_dict.pop('__version__')
    feat_dict.pop('__globals__')
    if feat_dict == {}:
        is_reproduction = True
        warnings.warn("Feat dict mat has nothing!! Reproduction is necessary!!")
        return is_reproduction, None
    cam_suffix = 'cam'
    label_suffix = 'label'
    list_img_suffix = 'list_img'
    for item in feat_dict.items():
        if item[0].endswith(cam_suffix):
            feat_dict[item[0]] = feat_dict[item[0]][0]
        if item[0].endswith(label_suffix):
            feat_dict[item[0]] = feat_dict[item[0]][0]
        if item[0].endswith(list_img_suffix):
            feat_dict[item[0]] = list(feat_dict[item[0]])
    return False, feat_dict
