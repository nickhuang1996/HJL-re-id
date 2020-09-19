import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class MDRSClassifier(nn.Module):
    def __init__(self, cfg):
        super(MDRSClassifier, self).__init__()
        num_classes = cfg.model.num_classes
        classifier = nn.Linear(cfg.model.em_dim, num_classes)
        self.cfg = cfg
        # classifier.apply(weights_init_classifier)
        classifier.apply(weights_init_kaiming)
        self.classifier = nn.ModuleList([copy.deepcopy(classifier) for _ in range(self.cfg.blocksetting.classifier)])

    def forward(self, out_dict):
        reduction_pool_feat_list = out_dict['reduction_pool_feat_list']
        cls_feat_list = []
        for i in range(len(self.classifier)):
            cls_feat_list.append(self.classifier[i](reduction_pool_feat_list[i]))

        out_dict['cls_feat_list'] = cls_feat_list
        return out_dict
