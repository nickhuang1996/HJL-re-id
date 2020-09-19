from MDRSREID.Networks.create_backbone import create_backbone
import torch.nn as nn


class PCBBackbone(nn.Module):
    def __init__(self, cfg):
        super(PCBBackbone, self).__init__()
        # cfg.model.backbone.last_conv_stride = 1
        resnet = create_backbone(cfg)

        # resnet.layer4[0].downsample[0].stride = (1,1)
        # resnet.layer4[0].conv2.stride = (1,1)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, in_dict):
        features = self.backbone(in_dict['im'])

        in_dict['feat'] = features
        return in_dict
