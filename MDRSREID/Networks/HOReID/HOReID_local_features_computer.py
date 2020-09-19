import torch
import torch.nn as nn
import torch.nn.functional as F


class HOReIDLocalFeaturesComputer(nn.Module):
    def __init__(self, cfg):
        super(HOReIDLocalFeaturesComputer, self).__init__()
        self.cfg = cfg
        self.weight_global_feature = cfg.keypoints_model.weight_global_feature

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)

    def keypoints_confidence_normalize(self, keypoints_confidence, sc):
        """compute keypoints confidence, sc=13"""
        keypoints_confidence[:, sc:] = F.normalize(
            keypoints_confidence[:, sc:], 1, 1) * self.weight_global_feature  # global feature score_confidence
        keypoints_confidence[:, :sc] = F.normalize(keypoints_confidence[:, :sc], 1,
                                                   1) * self.weight_global_feature  # partial feature score_confidence
        return keypoints_confidence

    def forward(self, in_dict, out_dict):
        """
        the last one is global feature
        :param in_dict:
            - feat: [N, 2048, 16, 8]
        :param out_dict:
            - scoremap: [N, 12+1, 16, 8]
            - keypoints_confidence: [N, 12+1]
        :return: out_dict:
            - feat_vec_list: 14 * [N, 2048]
            - keypoints_confidence: [N, 12+1+1]
        """
        feat = in_dict['feat']  # [N, 2048, 16, 8]
        scoremap = out_dict['scoremap']  # [N, 12+1, 16, 8]
        keypoints_confidence = out_dict['keypoints_condidence']  # [N, 13]

        fbs, fc, fh, fw = feat.shape
        sbs, sc, sh, sw = scoremap.shape
        assert fbs == sbs and fh == sh and fw == sw

        # get feat_vec_list
        feat_vec_list = []
        for i in range(sc + 1):
            if i < sc:  # skeleton-based local feature vectors
                score_map_i = scoremap[:, i, :, :].unsqueeze(1).repeat([1, fc, 1, 1])  # [N, 1, 16, 8] ==> [N, 2048, 16, 8]
                feat_vec_i = torch.sum(score_map_i * feat, [2, 3])  # [N, 2048]
                feat_vec_list.append(feat_vec_i)
            else:  # global feature vectors
                feat_vec_i = (self.GAP(feat) + self.GMP(feat)).squeeze()  # [N, 2048]
                feat_vec_list.append(feat_vec_i)
                keypoints_confidence = torch.cat([keypoints_confidence, torch.ones([fbs, 1]).cuda()],dim=1)  # [N, 13+1]

        keypoints_confidence = self.keypoints_confidence_normalize(keypoints_confidence, sc)  # [N, 13] ==> [N, 14]
        out_dict['feat_vec_list'] = feat_vec_list  # 14*[N, 2048]
        out_dict['keypoints_confidence'] = keypoints_confidence  # [N, 14]
        return out_dict

