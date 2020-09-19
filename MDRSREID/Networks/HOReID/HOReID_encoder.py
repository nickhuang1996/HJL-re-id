import torch.nn as nn
from MDRSREID.Networks.create_backbone import create_backbone


class HOReIDEncoder(nn.Module):
    def __init__(self, cfg):
        super(HOReIDEncoder, self).__init__()
        self.cfg = cfg

        # backbone and optimize its architecture
        # cfg.model.backbone.last_conv_stride = 1
        resnet = create_backbone(cfg)

        # resnet.layer4[0].downsample[0].stride = (1,1)
        # resnet.layer4[0].conv2.stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,  # no relu
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, in_dict):
        features = self.resnet_conv(in_dict['im'])

        in_dict['feat'] = features
        return in_dict
