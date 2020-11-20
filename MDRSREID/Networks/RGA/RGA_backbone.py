from MDRSREID.Networks.create_backbone import create_backbone
from MDRSREID.Networks.RGA.RGA_module import RGAModule
import torch.nn as nn


class RGABackbone(nn.Module):
    def __init__(self, cfg):
        super(RGABackbone, self).__init__()
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

        self.height = cfg.dataset.im.h_w[0]
        self.width = cfg.dataset.im.h_w[1]
        self.channel_ratio = cfg.model.RGA.channel_ratio
        self.spatial_ratio = cfg.model.RGA.spatial_ratio
        self.downchannel_ratio = cfg.model.RGA.downchannel_ratio

        self.branch_name = cfg.model.RGA.branch_name  # 'rgasc'
        self.use_spatial_and_channel_dict = {
            'rgasc': {
                'use_spatial': True,
                'use_channel': True,
            },
            'rgas': {
                'use_spatial': True,
                'use_channel': False,
            },
            'rgac': {
                'use_spatial': False,
                'use_channel': True,
            }
        }

        height = self.height
        width = self.width

        self.RGAModules_dict = {
            '1': {
                'in_channel': 256,
                'in_spatial': (height // 4) * (width // 4),
            },
            '2': {
                'in_channel': 512,
                'in_spatial': (height // 8) * (width // 8),
            },
            '3': {
                'in_channel': 1024,
                'in_spatial': (height // 16) * (width // 16),
            },
            '4': {
                'in_channel': 2048,
                'in_spatial': (height // 16) * (width // 16),
            },
        }
        print("[RGA MODULES]:")
        for k, v in self.RGAModules_dict.items():
            print("rga_attention_module_{}'s information:".format(k))
            setattr(self,
                    'rga_attention_module_{}'.format(k),
                    RGAModule(in_channel=v['in_channel'],
                              in_spatial=v['in_spatial'],
                              use_spatial=self.use_spatial_and_channel_dict[self.branch_name]['use_spatial'],
                              use_channel=self.use_spatial_and_channel_dict[self.branch_name]['use_channel'],
                              channel_ratio=self.channel_ratio,  # 8
                              spatial_ratio=self.spatial_ratio,  # 8
                              downchannel_ratio=self.downchannel_ratio  # 8
                              ))

    def forward(self, in_dict):
        conv1_x = self.conv1(in_dict['im'])
        bn1_x = self.bn1(conv1_x)
        relu_x = self.relu(bn1_x)
        maxpool_x = self.maxpool(relu_x)

        layer_x_list = []
        rga_am_x_list = []
        for i in range(4):
            if i == 0:
                layer_x_list.append(getattr(self, 'layer{}'.format(i + 1))(maxpool_x))
                rga_am_x_list.append(getattr(self, 'rga_attention_module_{}'.format(i + 1))(layer_x_list[i]))
            else:
                layer_x_list.append(getattr(self, 'layer{}'.format(i + 1))(rga_am_x_list[i - 1]))
                rga_am_x_list.append(getattr(self, 'rga_attention_module_{}'.format(i + 1))(layer_x_list[i]))

        in_dict['feat'] = rga_am_x_list[-1]
        return in_dict
