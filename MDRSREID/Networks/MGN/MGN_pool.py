import torch.nn as nn


class MGNPool(nn.Module):
    def __init__(self, cfg):
        super(MGNPool, self).__init__()
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d if cfg.model.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d
        self.pool0 = self.pool((1, 1))
        self.pool1 = self.pool((2, 1))
        self.pool2 = self.pool((3, 1))

    def forward(self, in_dict):
        """
        :param in_dict['feat_list']
        :return: part_pool_feat_list:
            0: GP
            1: GP
            2: GP
            3: 1 PP
            4: 1 PP
            5: 2 PP
            6: 2 PP
            7: 2 PP
        """
        feat_list = in_dict['feat_list']
        pool_feat_list = []

        # Global Pool
        for i in range(3):
            pool_feat_list.append(eval('self.pool{}'.format(0))(feat_list[i]))

        # Part Pool
        for i in range(1, 3):
            temp_pool_feat = eval('self.pool{}'.format(i))(feat_list[i])
            for j in range(0, i + 1):
                pool_feat_list.append(temp_pool_feat[:, :, j:j+1, :])

        out_dict = {'pool_feat_list': pool_feat_list}
        return out_dict





