from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """Modified from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py"""
    def __init__(self, data_source, k=1):
        """
        :param data_source: dataset
        :param k: k images of each person
        """
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        # For each person id ,we choice k images.
        self.k = k
        self.index_dic = defaultdict(list)
        # Get each person images indices
        for index, sample in data_source.items.items():
            self.index_dic[sample['label']].append(index)
        # Get all person id
        self.pids = list(self.index_dic.keys())
        # The number of all person id
        self.num_pids = len(self.pids)

    def __len__(self):
        """
        :return: image number of all persons
        """
        return self.num_pids * self.k

    def __iter__(self):
        """
        :return: iter(ret) contains self.k * self.num_pids images
        """
        # Get the random indices [0, self.num_pids]
        indices = torch.randperm(self.num_pids)
        ret = []
        for i in indices:
            # Get the one person id
            pid = self.pids[i]
            # Get all index of this person id
            t = self.index_dic[pid]
            if len(t) >= self.k:
                t = np.random.choice(t, size=self.k, replace=False)
            else:
                t = np.random.choice(t, size=self.k, replace=True)
            ret.extend(t)
        return iter(ret)
