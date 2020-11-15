import torch.nn as nn


class PyramidalPool(nn.Module):
    def __init__(self, cfg):
        super(PyramidalPool, self).__init__()
        self.cfg = cfg
        self.num_parts = cfg.model.num_parts  # 6
        assert 24 % self.num_parts == 0
        self.basic_stripe_size = 24 // self.num_parts  # 4
        # [6, 5, 4, 3, 2, 1]
        self.num_in_each_level = [i for i in range(self.num_parts, 0, -1)]
        self.num_branches = sum(self.num_in_each_level)  # 6+5+4+3+2+1 = 21
        self.used_levels = cfg.model.used_levels

        self.avgpool = nn.AdaptiveAvgPool2d
        self.maxpool = nn.AdaptiveMaxPool2d
        self.avgpool_list = []
        self.maxpool_list = []
        for i in range(self.num_parts):
            self.avgpool_list.append(self.avgpool((1, 1)))
            self.maxpool_list.append(self.maxpool((1, 1)))

    def forward(self, in_dict):
        """
        :param in_dict['feat']
        :return: pool_feat_list:
        """
        feat = in_dict['feat']
        pool_feat_list = []
        idx_level = 0
        for idx_branch in range(self.num_branches):
            if idx_branch >= sum(self.num_in_each_level[0:idx_level + 1]):
                idx_level += 1

            if self.used_levels[idx_level] == 0:
                continue

            idx_in_each_level = idx_branch - sum(self.num_in_each_level[0: idx_level])

            # 4->8->12->16->20->24
            stripe_size_in_level = self.basic_stripe_size * (idx_level + 1)

            # [0,4]->[4, 8]->[8, 12]->[12, 16]->[16, 20]->[20, 24]->
            # [0,8]->[4, 12]->[8, 16]->[12, 20]->[16, 24]->
            # [0,12]->[4, 16]->[8, 20]->[12, 24]->
            # [0,16]->[4, 20]->[8, 24]->
            # [0,20]->[4, 24]->
            # [0,24]
            start = idx_in_each_level * self.basic_stripe_size
            end = start + stripe_size_in_level
            pool_feat_list.append(self.avgpool_list[idx_level](feat[:, :, start: end, :]) +
                                  self.maxpool_list[idx_level](feat[:, :, start: end, :]))

        out_dict = {'pool_feat_list': pool_feat_list}
        return out_dict





