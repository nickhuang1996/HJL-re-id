import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num, dropput):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num
        self.dropout = dropput

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        if self.dropout > 0:
            feature = self.drop(feature)
        cls_score = self.classifier(feature)
        return feature, cls_score


class RGABNClassifier(nn.Module):
    def __init__(self, cfg):
        super(RGABNClassifier, self).__init__()

        self.cfg = cfg

        self.in_dim = 2048
        self.class_num = cfg.model.num_classes
        self.dropout = 0.5
        self.classifier = BNClassifier(self.in_dim,
                                       self.class_num,
                                       self.dropout)

    def forward(self, out_dict):
        pool_feat = out_dict['pool_feat']

        bn_pool_feat, cls_bn_pool_feat = self.classifier(pool_feat)
        out_dict['bn_pool_feat'] = bn_pool_feat
        out_dict['cls_bn_pool_feat'] = cls_bn_pool_feat

        return out_dict

