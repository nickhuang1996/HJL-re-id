import torch.nn as nn
import torch


class PGFAPool(nn.Module):
    def __init__(self, cfg):
        super(PGFAPool, self).__init__()
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d if cfg.model.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d
        self.pool0 = self.pool((1, 1))
        self.pool1 = self.pool((cfg.model.num_parts, 1))

    def train_forward(self, pool_feat):
        return torch.squeeze(pool_feat)

    def test_forward(self, pool_feat, index):
        if index == 0:
            assert (pool_feat.shape[1], pool_feat.shape[2], pool_feat.shape[3]) == (2048, 1, 1), \
                "pool_feat.shape should be (2048, 1, 1), " \
                "but found ({}, {}, {})".format(pool_feat.shape[1],
                                                pool_feat.shape[2],
                                                pool_feat.shape[3])
            pool_feat = pool_feat.view(-1, 2048)
        elif index == 1:
            assert (pool_feat.shape[1], pool_feat.shape[2], pool_feat.shape[3]) == (2048, self.cfg.model.num_parts, 1), \
                "pool_feat.shape should be (2048, {}, 1), " \
                "but found ({}, {}, {})".format(self.cfg.model.num_parts,
                                                pool_feat.shape[1],
                                                pool_feat.shape[2],
                                                pool_feat.shape[3])
            pool_feat = torch.squeeze(pool_feat, -1)
            pool_feat = pool_feat.permute(0, 2, 1)
        return pool_feat

    def forward(self, in_dict, cfg):
        feat = in_dict['feat']

        pool_feat_list = []
        for i in range(2):
            pool_feat = eval('self.pool{}'.format(i))(feat)
            if cfg.model_flow is 'train':
                pool_feat = self.train_forward(pool_feat)
            elif cfg.model_flow is 'test':
                pool_feat = self.test_forward(pool_feat, index=i)
            else:
                raise ValueError('Invalid phase {}'.format(cfg.model_flow))
            pool_feat_list.append(pool_feat)

        out_dict = {
            'pool_feat_list': pool_feat_list,
        }
        return out_dict
