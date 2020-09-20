from MDRSREID.DataLoaders.Datasets import Dataset
from MDRSREID.utils.get_files_by_pattern import get_files_by_pattern
from MDRSREID.utils.walkdir import walkdir
import shutil
import os.path as osp
from PIL import Image
from scipy.misc import imsave
from tqdm import tqdm
import numpy as np
from MDRSREID.utils.may_make_dirs import may_make_dirs


class CUHK03NpDetectedJpg(Dataset):
    def __init__(self,
                 cfg=None,
                 mode=None,
                 domain=None,
                 name=None,
                 authority=None,
                 train_type=None,
                 items=None):
        super(CUHK03NpDetectedJpg, self).__init__(cfg)
        self.cfg = cfg
        self.num_cam = 2
        self.train_type = train_type  # Supervised or Unsupervised
        self.mode = mode  # for transform
        self.domain = domain
        self.dataset_root = osp.join(cfg.dataset.root, name)
        self.authority = authority
        self.png_im_root = 'cuhk03-np'
        self.im_root = 'cuhk03-np-jpg'
        # If you don't have jpg files, please use this function
        # self._save_png_as_jpg()

        self.train_dir = osp.join(self.im_root, 'bounding_box_train')
        self.query_dir = osp.join(self.im_root, 'query')
        self.gallery_dir = osp.join(self.im_root, 'bounding_box_test')
        self.im_authority = {
            'train': {
                'dir': self.train_dir,
                'pattern': '{}/detected/bounding_box_train/*.jpg'.format(self.im_root),
                'map_label': True},
            'query': {
                'dir': self.query_dir,
                'pattern': '{}/detected/query/*.jpg'.format(self.im_root),
                'map_label': False},
            'gallery': {
                'dir': self.gallery_dir,
                'pattern': '{}/detected/bounding_box_test/*.jpg'.format(self.im_root),
                'map_label': False},
        }
        # all im path in a list
        self.im_path = sorted(get_files_by_pattern(root=self.dataset_root,
                                                   pattern=self.im_authority[self.authority]['pattern'],
                                                   strip_root=True)
                              )
        # Filter out -1
        self.im_path = [self.im_path[i] for i in range(len(self.im_path))]
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        if items is None:
            self.items = self.get_items()

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.replace('\\', '/').split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.replace('\\', '/').split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.im_path]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def num_ids(self):
        """
        :return: unique person ids number
        """
        return len(self.unique_ids)

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.im_path]

    def get_items(self):
        """
        :return: get the items:
            'im_path':
            'label':
            'cam':

        The label may be index or not by 'map_label'
        """
        if self.im_authority[self.authority]['map_label'] is False:
            items = {
                i:
                    {
                        'im_path': self.im_path[i],
                        'label': self.id(self.im_path[i]),
                        'cam': self.camera(self.im_path[i])
                    }
                for i in range(len(self.im_path))}
        else:
            items = {
                i:
                    {
                        'im_path': self.im_path[i],
                        'label': self._id2label[self.id(self.im_path[i])],
                        'cam': self.camera(self.im_path[i])
                    }
                for i in range(len(self.im_path))}
        return items

    def _get_ps_label_path(self, im_path):
        """
        :param im_path: the ps_label path
        :return:
        """
        path = im_path.replace(self.im_root, self.im_root + '_ps_label').replace('.jpg', '.png')
        path = osp.join(self.dataset_root, path)
        return path

    def _check_before_get_im_path(self):
        """
        :return:
        """
        self._check_dir = osp.join(self.dataset_root, self.im_authority[self.authority]['dir']).replace('\\', '/')
        if not osp.exists(self.dataset_root):
            raise RuntimeError("'{}' is not available".format(self.dataset_root))
        if not osp.exists(self._check_dir):
            raise RuntimeError("'{}' is not available".format(self._check_dir))
        print("check {} dataset success!!".format(self.authority))

    def _save_png_as_jpg(self):
        png_im_dir = osp.join(self.dataset_root, self.png_im_root)
        jpg_im_dir = osp.join(self.dataset_root, self.im_root)
        assert osp.exists(png_im_dir), "The PNG image dir {} should be placed inside {}".format(png_im_dir, self.dataset_root)
        png_paths = list(walkdir(png_im_dir, '.png'))
        if osp.exists(jpg_im_dir):
            jpg_paths = list(walkdir(jpg_im_dir, '.jpg'))
            if len(png_paths) == len(jpg_paths):
                print('=> Found same number of JPG images. Skip transforming PNG to JPG.')
                return
            else:
                shutil.rmtree(jpg_im_dir)
                print('=> JPG image dir exists but does not contain same number of images. So it is removed and would be re-generated.')
        # CUHK03 contains 14097 detected + 14096 labeled images = 28193
        print('change dir is :', osp.dirname(png_paths[0]))
        for png_path in tqdm(png_paths, desc='PNG->JPG', miniters=2000, ncols=120, unit=' images'):
            jpg_path = png_path.replace(self.png_im_root, self.im_root).replace('.png', '.jpg')
            may_make_dirs(dst_path=osp.dirname(jpg_path))
            imsave(jpg_path, np.array(Image.open(png_path)))
