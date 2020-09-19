import torch.nn as nn
import torch
import copy
from ..weights_init import weights_init_kaiming


class MDRSReduction(nn.Module):
    def __init__(self, cfg):
        super(MDRSReduction, self).__init__()
        self.cfg = cfg
        reduction = nn.Sequential(nn.Conv2d(2048, cfg.model.em_dim, 1, bias=False),
                                  nn.BatchNorm2d(cfg.model.em_dim),
                                  nn.ReLU()
                                  )

        reduction.apply(weights_init_kaiming)
        self.reduction = nn.ModuleList([copy.deepcopy(reduction) for _ in range(self.cfg.blocksetting.reduction)])

    def forward(self, out_dict):
        pool_feat_list = out_dict['pool_feat_list']

        reduction_pool_feat_list = []
        for i in range(len(self.reduction)):
            reduction_pool_feat_list.append(self.reduction[i](pool_feat_list[i]).squeeze(dim=3).squeeze(dim=2))

        out_dict['reduction_pool_feat_list'] = reduction_pool_feat_list
        if self.cfg.blocksetting.backbone != 1:
            triplet_reduction_pool_feat = torch.cat(reduction_pool_feat_list[:self.cfg.blocksetting.backbone], 1)
        else:
            triplet_reduction_pool_feat = reduction_pool_feat_list[0]
        out_dict['triplet_reduction_pool_feat'] = triplet_reduction_pool_feat
        return out_dict
