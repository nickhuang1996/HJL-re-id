import torch.nn as nn


class PCBPool(nn.Module):
    def __init__(self, cfg):
        super(PCBPool, self).__init__()
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d if cfg.model.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d
        self.pool0 = self.pool((cfg.model.num_parts, 1))

    def forward(self, in_dict):
        feat = in_dict['feat']

        pool_feat = self.pool0(feat)

        out_dict = {
            'pool_feat': pool_feat,
        }
        return out_dict





