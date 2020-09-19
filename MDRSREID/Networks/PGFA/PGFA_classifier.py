import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class PGFAClassifier(nn.Module):
    def __init__(self, cfg):
        super(PGFAClassifier, self).__init__()
        num_classes = cfg.model.num_classes

        classifier = []
        classifier += [nn.Linear(cfg.model.em_dim, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = nn.ModuleList([copy.deepcopy(classifier) for _ in range(cfg.model.num_parts + 1)])

    def forward(self, out_dict, cfg):
        reduction_pool_feat_list = out_dict['reduction_pool_feat_list']

        cls_feat_list = []
        if cfg.model_flow is 'train':
            for i in range(len(self.classifier)):
                cls_feat_list.append(self.classifier[i](reduction_pool_feat_list[i]))
        elif cfg.model_flow is 'test':
            cls_feat_list.append(self.classifier[0](reduction_pool_feat_list[0]))
        else:
            raise ValueError('Invalid phase {}'.format(cfg.model_flow))

        out_dict['cls_feat_list'] = cls_feat_list
        return out_dict
