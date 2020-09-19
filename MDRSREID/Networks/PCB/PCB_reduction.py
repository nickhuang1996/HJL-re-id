import torch.nn as nn
import copy
from ..weights_init import weights_init_kaiming


class PCBReduction(nn.Module):
    def __init__(self, cfg):
        super(PCBReduction, self).__init__()
        self.dropout = nn.Dropout(0.5)
        reduction = []
        reduction.append(nn.Conv2d(2048, cfg.model.em_dim, 1, stride=1, padding=0, bias=False))
        reduction.append(nn.BatchNorm2d(cfg.model.em_dim))

        if cfg.model.reduction.use_relu == 'relu':
            reduction.append(nn.ReLU(inplace=True))
        elif cfg.model.reduction.use_leakyrelu == 'leakyrelu':
            reduction.append(nn.LeakyReLU(0.1))

        self.reduction = nn.Sequential(*reduction)
        self.reduction.apply(weights_init_kaiming)

    def forward(self, out_dict):
        pool_feat = out_dict['pool_feat']
        reduction_pool_feat = self.reduction(pool_feat).squeeze(dim=3).squeeze(dim=2)

        out_dict['reduction_pool_feat'] = reduction_pool_feat
        return out_dict
