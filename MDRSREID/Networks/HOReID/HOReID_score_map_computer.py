import torch
import torch.nn as nn
import torch.nn.functional as F

from MDRSREID.Networks.create_keypoints_predictor import create_keypoints_predictor
from MDRSREID.Networks.HOReID.HeatmapProcessor import HeatmapProcessor2


class HOReIDScoreMapComputer(nn.Module):
    def __init__(self, cfg):
        super(HOReIDScoreMapComputer, self).__init__()
        self.cfg = cfg

        # init skeletion model
        self.keypoints_predictor = create_keypoints_predictor(cfg)
        self.heatmap_processor = HeatmapProcessor2(cfg=cfg,
                                                   normalize_heatmap=True,
                                                   group_mode='sum',
                                                   norm_scale=cfg.keypoints_model.norm_scale)

    def forward(self, in_dict):
        """
        :param x: inputs from encoder(ResNet) [N, 2048, 16, 8]
        :return:
            scoremap: [N, 12+1, 16, 8]
            keypoints_condidence: [N, 12+1]
            keypoints_location: [N, 17, 2]
        """
        heatmap = self.keypoints_predictor(in_dict['im'])  # [N, 2048, 16, 8] ==> [N, 17, 64, 32]
        out_dict = {}
        out_dict['label'] = in_dict['label']
        out_dict['heatmap'] = heatmap
        scoremap, keypoints_condidence, keypoints_location = self.heatmap_processor(heatmap) # [N, 12+1, 16, 8] [N, 12+1] [N, 17, 2]
        out_dict['scoremap'] = scoremap.detach()
        out_dict['keypoints_condidence'] = keypoints_condidence.detach()
        out_dict['keypoints_location'] = keypoints_location.detach()
        return out_dict
