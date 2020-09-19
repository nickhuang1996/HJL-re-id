import torch
import torch.nn as nn


class PGFAPoseGuidedMaskBlock(nn.Module):
    def __init__(self, cfg):
        super(PGFAPoseGuidedMaskBlock, self).__init__()
        self.cfg = cfg

        self.pg_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pg_maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, in_dict, out_dict, cfg):
        pool_global_feat = out_dict['pool_feat_list'][0]
        masks = in_dict['pose_landmark_mask']
        feat = in_dict['feat']  # global feature

        if cfg.model_flow is 'train':
            temp_pg_global_feat = torch.cuda.FloatTensor()
        elif cfg.model_flow is 'test':
            temp_pg_global_feat = []
        else:
            raise ValueError('Invalid phase {}'.format(cfg.model_flow))
        for i in range(18):  # There are 18 pose landmarks per image
            mask = masks[:, i, :, :]
            mask = torch.unsqueeze(mask, 1)
            mask = mask.expand_as(feat)

            pg_feat_ = mask * feat  # element-wise multiplication
            pg_feat_ = self.pg_avgpool(pg_feat_)
            if cfg.model_flow is 'train':
                pg_feat_ = torch.squeeze(pg_feat_)
                pg_feat_ = torch.unsqueeze(pg_feat_, 2)
                temp_pg_global_feat = torch.cat((temp_pg_global_feat, pg_feat_), 2)
            elif cfg.model_flow is 'test':
                pg_feat_ = pg_feat_.view(-1, 2048, 1)
                temp_pg_global_feat.append(pg_feat_)
            else:
                raise ValueError('Invalid phase {}'.format(cfg.model_flow))
        if cfg.model_flow is 'test':
            temp_pg_global_feat = torch.cat((temp_pg_global_feat), 2)
        temp_pg_global_feat = self.pg_maxpool(temp_pg_global_feat)
        if cfg.model_flow is 'train':
            temp_pg_global_feat = torch.squeeze(temp_pg_global_feat)
        else:
            temp_pg_global_feat = temp_pg_global_feat.view(-1, 2048)
        pg_global_feat = torch.cat((pool_global_feat, temp_pg_global_feat), 1)

        out_dict['pool_feat_list'][0] = pg_global_feat
        return out_dict
