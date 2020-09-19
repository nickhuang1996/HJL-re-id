from MDRSREID.Networks.create_backbone import create_backbone
from MDRSREID.utils.load_state_dict import load_state_dict
from MDRSREID.Networks.RESIDUAL_NETWORK.RESNET.resnet import Bottleneck
import torch.nn as nn
import copy


class MDRSBackbone(nn.Module):
    def __init__(self, cfg):
        super(MDRSBackbone, self).__init__()
        resnet = create_backbone(cfg)
        self.cfg = cfg
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )
        self.out_c = resnet.out_c
        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        global_conv5 = resnet.layer4

        # Option 1
        # local_conv5_1 = nn.Sequential(
        #     Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, 2, bias=False), nn.BatchNorm2d(2048))),
        #     Bottleneck(2048, 512),
        #     Bottleneck(2048, 512))
        #
        # load_state_dict(local_conv5_1, resnet.layer4.state_dict())
        #
        # local_conv5_2 = nn.Sequential(
        #     Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
        #     Bottleneck(2048, 512),
        #     Bottleneck(2048, 512))
        #
        # load_state_dict(res_p_conv5_2, resnet.layer4.state_dict())
        #
        # self.p0 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(global_conv5))
        # self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(local_conv5_1))
        # self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(local_conv5_2))

        # Option 2
        local_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        local_conv5.load_state_dict(resnet.layer4.state_dict())
        self.p0 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(global_conv5))
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(local_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(local_conv5))

    def forward(self, in_dict):
        features = self.backbone(in_dict['im'])

        part_feat_list = []

        for i in range(self.cfg.blocksetting.backbone):
            part_feat_list.append(eval('self.p{}'.format(i))(features))
        in_dict['feat_list'] = part_feat_list
        return in_dict
