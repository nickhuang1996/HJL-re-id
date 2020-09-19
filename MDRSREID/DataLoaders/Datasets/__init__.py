from torch.utils.data import Dataset as TorchDataset
from MDRSREID.utils.data_utils.transforms.torch_transforms import transforms
from copy import deepcopy
import os.path as osp
from PIL import Image
import numpy as np
import torch


class Dataset(TorchDataset):
    """
    TorchDataset has 3 abstract methods:
        __init__
        __getitem__
        __len__
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_root = None
        self.mode = None
        self.items = None
        self.num_cam = None
        self.train_type = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cfg = self.cfg
        mode = self.mode # train or test
        # Deepcopy to inherit all meta items
        item = deepcopy(self.items[index])
        if self.train_type == 'Supervised':
            im_path = item['im_path']
            if cfg.dataset.use_ps_label:
                item['ps_label'] = self.get_ps_label(item['im_path'])
            if cfg.dataset.use_pose_landmark_mask:
                item['pose_landmark_mask'] = self.get_pose_landmark_mask(item['im_path'])
            if self.mode == 'test' and cfg.dataset.use_occlude_duke is True:
                item['test_pose_path'] = self.get_test_pose_path(item['im_path'])
        elif self.train_type == 'Unsupervised':
            im_path = self.get_camstyle_im(item)
            item['index'] = index
        else:
            im_path = None
        item['im'] = self.get_im(im_path)
        item['width'], item['height'] = item['im'].size
        # if cfg.dataset.use_ps_label:
        #     item['ps_label'] = self.get_ps_label(item['im_path'])
        transforms(item, cfg, mode)
        item['im_path'] = osp.join(self.dataset_root, im_path)
        return item

    def get_im(self, im_path):
        """
        :param im_path:
        :return: a PIL image, which has been converted to RGB(PNG file has 4 channels RGBA)
        """
        im = Image.open(osp.join(self.dataset_root, im_path)).convert("RGB")
        return im

    def _get_ps_label_path(self, im_path):
        raise NotImplementedError

    def get_ps_label(self, im_path):
        ps_label = Image.open(self._get_ps_label_path(im_path))
        return ps_label

    def _get_pose_landmark_mask(self, im_path):
        raise NotImplementedError

    def get_pose_landmark_mask(self, im_path):
        pose_landmark_mask = np.load(self._get_pose_landmark_mask(im_path))
        return pose_landmark_mask

    def _get_test_pose_path(self, im_path):
        raise NotImplementedError

    def get_test_pose_path(self, im_path):
        """
        :param im_path:
        :return:

        For only self.mode is 'test'
        """
        test_pose_path = self._get_test_pose_path(im_path)
        return test_pose_path

    def get_camstyle_im(self, item):
        sel_cam = torch.randperm(self.num_cam)[0] + 1  # [1, n]
        if sel_cam == item['cam']:
            im_path = item['im_path']
        else:
            im_path_split_list = item['im_path'].replace('\\', '/').split('/')
            im_basename = im_path_split_list[-1][:-4] + '_fake_' + str(item['cam']) + 'to' + str(sel_cam.numpy()) + '.jpg'
            im_camstyle_dir = im_path_split_list[-3] + '/' + im_path_split_list[-2] + '_camstyle'
            im_path = osp.join(im_camstyle_dir, im_basename)
        return im_path

    def _check_before_get_im_path(self):
        raise NotImplementedError
