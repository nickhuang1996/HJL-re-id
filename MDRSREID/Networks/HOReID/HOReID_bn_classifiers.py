import torch.nn as nn
from ..weights_init import weights_init_classifier, weights_init_kaiming
import copy


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score


class HOReIDBNClassifiers(nn.Module):
    def __init__(self, cfg):
        super(HOReIDBNClassifiers, self).__init__()

        self.cfg = cfg

        self.in_dim = 2048
        self.class_num = cfg.model.num_classes
        self.branch_num = cfg.keypoints_model.branch_num
        self._check_branch_num()

        for i in range(self.branch_num):
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.in_dim, self.class_num))

    def _check_branch_num(self):
        """
        This function is to check the branch number followed by:
            len(joints_groups) + 1 = branch_num
        where `1` is donated as the global feature.
        """
        assert len(self.cfg.keypoints_model.joints_groups) + 1 == self.cfg.keypoints_model.branch_num, \
            "BNClassifiers branch numbers must be {}! The global feature is supposed to be considered!". \
            format(len(self.cfg.keypoints_model.joints_groups) + 1)

    def __call__(self, out_dict, is_gcned):

        if is_gcned:
            feat_vec_list = out_dict['gcned_feat_vec_list']
        else:
            feat_vec_list = out_dict['feat_vec_list']

        assert len(feat_vec_list) == self.branch_num

        # bnneck for each sub_branch_feature
        bned_feat_vec_list = []
        cls_score_list = []
        for i in range(self.branch_num):
            feat_vec_i = feat_vec_list[i]

            classifier_i = getattr(self, 'classifier_{}'.format(i))
            bned_feat_vec_i, cls_score_i = classifier_i(feat_vec_i)

            bned_feat_vec_list.append(bned_feat_vec_i)
            cls_score_list.append(cls_score_i)

        if is_gcned:
            out_dict['bned_gcned_feat_vec_list'] = bned_feat_vec_list
            out_dict['gcned_cls_score_list'] = cls_score_list
        else:
            out_dict['bned_feat_vec_list'] = bned_feat_vec_list
            out_dict['cls_score_list'] = cls_score_list
        return out_dict


