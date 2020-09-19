import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class PGFAReduction(nn.Module):
    def __init__(self, cfg):
        super(PGFAReduction, self).__init__()
        self.cfg = cfg

        global_block = [nn.Dropout(0.5)]
        part_feat_block = [nn.Dropout(0.5)]
        global_block += [nn.Linear(cfg.model.PGFA.global_input_dim, cfg.model.em_dim)]
        part_feat_block += [nn.Linear(cfg.model.PGFA.part_feat_input_dim, cfg.model.em_dim)]

        add_block = []
        add_block += [nn.BatchNorm1d(cfg.model.em_dim)]

        if cfg.model.reduction.use_relu:
            add_block += [nn.LeakyReLU(0.1)]
        if cfg.model.reduction.use_dropout:
            add_block += [nn.Dropout(0.5)]
        global_reduction = nn.Sequential(*global_block, *add_block)
        part_feat_reduction = nn.Sequential(*part_feat_block, *add_block)
        global_reduction.apply(weights_init_kaiming)
        part_feat_reduction.apply(weights_init_kaiming)

        self.reduction = nn.ModuleList([global_reduction, part_feat_reduction])

    def forward(self, out_dict, cfg):
        pool_feat_list = out_dict['pool_feat_list']

        reduction_pool_feat_list = []
        for i in range(len(self.reduction)):
            if i == 0:  # Pose Guided Global Feature Branch
                reduction_pool_feat_list.append(self.reduction[i](pool_feat_list[i]))
            else:       # Partial Feature Branch
                if cfg.model_flow is 'train':
                    for j in range(self.cfg.model.num_parts):
                        reduction_pool_feat_list.append(self.reduction[i](pool_feat_list[i][:, :, j]))

        out_dict['reduction_pool_feat_list'] = reduction_pool_feat_list
        return out_dict
