import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class PyramidalClassifier(nn.Module):
    def __init__(self, cfg):
        super(PyramidalClassifier, self).__init__()
        num_classes = cfg.model.num_classes
        classifier = nn.Linear(cfg.model.em_dim, num_classes)

        classifier.apply(weights_init_classifier)
        # classifier.apply(weights_init_kaiming)
        self.classifier = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.2)

        self.num_parts = cfg.model.num_parts  # 6
        self.used_levels = cfg.model.used_levels
        # [6, 5, 4, 3, 2, 1]
        self.num_in_each_level = [i for i in range(self.num_parts, 0, -1)]

        self.num_levels = len(self.num_in_each_level)  # 6
        self.num_branches = sum(self.num_in_each_level)  # 6+5+4+3+2+1 = 21

        idx_level = 0
        for idx_branch in range(self.num_branches):
            if idx_branch >= sum(self.num_in_each_level[0: idx_level + 1]):
                idx_level += 1

            if self.used_levels[idx_level] == 0:
                continue

            self.classifier.append(copy.deepcopy(classifier))

    def forward(self, out_dict):
        reduction_pool_feat_list = out_dict['reduction_pool_feat_list']
        cls_feat_list = []
        for i in range(len(self.classifier)):
            cls_feat_list.append(self.classifier[i](self.dropout(reduction_pool_feat_list[i])))

        out_dict['cls_feat_list'] = cls_feat_list
        return out_dict
