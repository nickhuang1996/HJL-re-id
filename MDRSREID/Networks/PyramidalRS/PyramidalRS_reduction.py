import torch.nn as nn
import copy
from ..weights_init import weights_init_kaiming


class PyramidalRSReduction(nn.Module):
    def __init__(self, cfg):
        super(PyramidalRSReduction, self).__init__()
        # em_dim = 128
        reduction = nn.Sequential(nn.Conv2d(2048, cfg.model.em_dim, 1, bias=True),
                                  nn.BatchNorm2d(cfg.model.em_dim),
                                  nn.ReLU(inplace=True)
                                  )

        reduction.apply(weights_init_kaiming)
        self.reduction = nn.ModuleList()

        self.num_parts = cfg.model.num_parts  # 6
        self.used_levels = cfg.model.used_levels
        # [6, 5, 4, 3, 2, 1]
        self.num_in_each_level = [i for i in range(self.num_parts, 0, -1)]

        self.num_levels = len(self.num_in_each_level) # 6
        self.num_branches = sum(self.num_in_each_level) # 6+5+4+3+2+1 = 21

        idx_level = 0
        for idx_branch in range(self.num_branches):
            if idx_branch >= sum(self.num_in_each_level[0: idx_level + 1]):
                idx_level += 1

            if self.used_levels[idx_level] == 0:
                continue

            self.reduction.append(copy.deepcopy(reduction))

    def forward(self, out_dict):
        pool_feat_list = out_dict['pool_feat_list']
        reduction_pool_feat_list = []
        for i in range(len(self.reduction)):
            reduction_pool_feat_list.append(self.reduction[i](pool_feat_list[i]).squeeze(dim=3).squeeze(dim=2))

        out_dict['reduction_pool_feat_list'] = reduction_pool_feat_list
        triplet_reduction_pool_feat = reduction_pool_feat_list[-1]
        out_dict['triplet_reduction_pool_feat'] = triplet_reduction_pool_feat
        return out_dict
