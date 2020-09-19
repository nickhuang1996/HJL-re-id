import torch.nn as nn
import copy
from ..weights_init import weights_init_kaiming


class MGNReduction(nn.Module):
    def __init__(self, cfg):
        super(MGNReduction, self).__init__()

        reduction = nn.Sequential(nn.Conv2d(2048, cfg.model.em_dim, 1, bias=False),
                                  nn.BatchNorm2d(cfg.model.em_dim),
                                  nn.ReLU()
                                  )

        reduction.apply(weights_init_kaiming)
        self.reduction = nn.ModuleList([copy.deepcopy(reduction) for _ in range(8)])

    def forward(self, out_dict):
        pool_feat_list = out_dict['pool_feat_list']
        reduction_pool_feat_list = []
        for i in range(len(self.reduction)):
            reduction_pool_feat_list.append(self.reduction[i](pool_feat_list[i]).squeeze(dim=3).squeeze(dim=2))

        out_dict['reduction_pool_feat_list'] = reduction_pool_feat_list
        return out_dict
