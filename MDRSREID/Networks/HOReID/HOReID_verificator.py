import torch.nn as nn
import torch
import torch.nn.functional as F


class HOReIDVerificator(nn.Module):
    def __init__(self, cfg):
        super(HOReIDVerificator, self).__init__()

        self.cfg = cfg
        self.branch_num = cfg.keypoints_model.branch_num

        self.bn = nn.BatchNorm1d(2048 * self.branch_num)  # 2048 * 14
        self.layer1 = nn.Linear(2048 * self.branch_num, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _verificator(self, feat_vec_list1, feat_vec_list2):
        """
        :param feat_vec_list1: list with length node_num, element size is [N, feature_length]
        :param feat_vec_list2: list with length node_num, element size is [N, feature_length]
        :return:
        """
        if type(feat_vec_list1).__name__ == 'list':
            cated_feat_vec_1 = torch.cat([feat_vec_1.unsqueeze(1) for feat_vec_1 in feat_vec_list1], dim=1)
            cated_feat_vec_2 = torch.cat([feat_vec_2.unsqueeze(1) for feat_vec_2 in feat_vec_list2], dim=1)
        elif type(feat_vec_list1).__name__ == 'Tensor':  # [N, branch_num, c]
            cated_feat_vec_1 = feat_vec_list1
            cated_feat_vec_2 = feat_vec_list2

        # cated_feat_vec_1 = cated_feat_vec_1.detach()
        # cated_feat_vec_2 = cated_feat_vec_2.detach()

        # Normalize
        normalize_cated_feat_vec_1 = F.normalize(cated_feat_vec_1, p=2, dim=2)
        normalize_cated_feat_vec_2 = F.normalize(cated_feat_vec_2, p=2, dim=2)

        feat = self.cfg.eval.ver_in_scale * \
            normalize_cated_feat_vec_1 * \
            normalize_cated_feat_vec_2
        feat = feat.view([feat.shape[0], feat.shape[1] * feat.shape[2]])

        logit = self.layer1(feat)
        prob = self.sigmoid(logit)

        return prob

    def __call__(self, out_dict, cfg):

        if cfg.stage is 'FeatureExtract':
            emb_prefix = 'bned_gcned_feat_vec_'
            emb_list = ['pos', 'neg']
            prob_list = []
            for i in range(len(emb_list)):
                feat_vec_list1 = out_dict[emb_prefix + emb_list[i]]
                feat_vec_list2 = out_dict[emb_prefix + emb_list[i] + '_' + emb_list[i]]
                prob = self._verificator(feat_vec_list1=feat_vec_list1,
                                         feat_vec_list2=feat_vec_list2)
                prob_list.append(prob)
            out_dict['ver_prob_pos'] = prob_list[0]
            out_dict['ver_prob_neg'] = prob_list[1]
        elif cfg.stage is 'Evaluation':
            a_sample_query_feat_stage2 = out_dict['a_sample_query_feat_stage2']
            topk_gallery_feat_stage2 = out_dict['topk_gallery_feat_stage2']
            prob = self._verificator(feat_vec_list1=a_sample_query_feat_stage2,
                                     feat_vec_list2=topk_gallery_feat_stage2)
            out_dict['ver_prob'] = prob

        return out_dict
