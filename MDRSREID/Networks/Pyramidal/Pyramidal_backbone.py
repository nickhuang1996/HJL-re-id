from MDRSREID.Networks.create_backbone import create_backbone
import torch.nn as nn


class PyramidalBackbone(nn.Module):
    def __init__(self, cfg):
        super(PyramidalBackbone, self).__init__()

        resnet = create_backbone(cfg)

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
