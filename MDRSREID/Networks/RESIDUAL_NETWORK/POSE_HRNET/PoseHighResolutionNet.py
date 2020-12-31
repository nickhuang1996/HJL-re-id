import torch
import torch.nn as nn
from MDRSREID.Networks.RESIDUAL_NETWORK.convolution_layers import conv1x1
from MDRSREID.Networks.RESIDUAL_NETWORK.BasicBlock import BasicBlock
from MDRSREID.Networks.RESIDUAL_NETWORK.Bottleneck import Bottleneck
from MDRSREID.Networks.RESIDUAL_NETWORK.POSE_HRNET.HighResolutionModule import HighResolutionModule

import os.path as osp

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.keypoints_model.extra  # model_extras['pose_high_resolution_net']
        super(PoseHighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(blocks_dict['BOTTLENECK'], 64, 4)  # 4 Bottlenecks, channels 64 --> 256

        self.stage_cfg_list = []
        self.transition_list = []
        self.stage_list = []
        self.stage_multi_scale_output_dict = {
            '2': True,
            '3': True,
            '4': False
        }
        pre_stage_channels = [256]
        for i in range(3):
            self.stage_cfg_list.append(eval('cfg.keypoints_model.extra.stage{}'.format(i + 2)))
            current_stage_cfg = self.stage_cfg_list[i]
            block = blocks_dict[current_stage_cfg['block']]  # block type, Basic or Bottleneck

            num_channels = current_stage_cfg['num_channels']  # [48, 96]
            num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # output_channels = input_channels * 4

            self.transition_list.append(self._make_transition_layer(num_channels_pre_layer=pre_stage_channels,  # transition123
                                                                    num_channels_cur_layer=num_channels))
            current_stage, pre_stage_channels = self._make_stage(stage_cfg=current_stage_cfg,  # stage234, HighResolutionModule
                                                                 num_inchannels=num_channels,
                                                                 multi_scale_output=self.stage_multi_scale_output_dict['{}'.format(i + 2)])
            self.stage_list.append(current_stage)
        self.transition1 = self.transition_list[0]
        self.transition2 = self.transition_list[1]
        self.transition3 = self.transition_list[2]

        self.stage2 = self.stage_list[0]
        self.stage3 = self.stage_list[1]
        self.stage4 = self.stage_list[2]

        # classifier
        self.final_layer = nn.Conv2d(  # size is not changed
            pre_stage_channels[0],
            cfg.keypoints_model.num_joints,  # 17
            kernel_size=extra.final_conv_kernel,
            stride=1,
            padding=1 if extra.final_conv_kernel == 3 else 0
        )

        self.pretrained_layers = cfg.keypoints_model.extra.pretrained_layers # '*'

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        This function is used for `layer1`.
        However, in `ResNet50`, layer1's blocks is 3, not 4.
        :param block: Bottleneck
        :param planes: 64
        :param blocks: 4
        :param stride: 1
        :return: layer1
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=stride, bias=False),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition_layer(num_channels_pre_layer,
                               num_channels_cur_layer):
        """
        :param num_channels_pre_layer:
        :param num_channels_cur_layer:
        :return: transition_layers_list
        This function:
            [0, len(num_channels_pre_layer)] : feature maps' size is not changed
            [len(num_channels_pre_layer), len(num_channels_cur_layer)] : feature maps' size is half step by step, downsample
        """
        transition_layers_list = []
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers_list.append(
                        nn.Sequential(
                            nn.Conv2d(  # size is not changed.
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers_list.append(None)
            else:
                Conv3x3s_list = []
                for j in range(i + 1 - len(num_channels_pre_layer)):
                    inchannels = num_channels_pre_layer[-1]  # the last one
                    outchannels = num_channels_cur_layer[i] if j == i - len(num_channels_pre_layer) else inchannels
                    Conv3x3s_list.append(
                        nn.Sequential(
                            nn.Conv2d(  # size = 0.5 * size
                                inchannels,
                                outchannels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers_list.append(nn.Sequential(*Conv3x3s_list))

        return nn.ModuleList(transition_layers_list)

    def _make_stage(self,
                    stage_cfg,
                    num_inchannels,
                    multi_scale_output=True):
        """
        BRANCH AND FUSE
        :param stage_cfg: after layer1, stage234
        :param num_inchannels: use in `HighResolutionModule`
        :param multi_scale_output: Use in making `HighResolutionModule`s
        :return: nn.Sequential(*modules_list) --> stage, and `num_inchannels`

        **Note**:
            num_inchannels[branch_index] = \
                num_channels[branch_index] * block.expansion
        This is for next module's `num_inchannels`
        """
        num_modules = stage_cfg['num_modules']
        num_branches = stage_cfg['num_branches']
        num_blocks = stage_cfg['num_blocks']
        num_channels = stage_cfg['num_channels']
        block = blocks_dict[stage_cfg['block']]
        fuse_method = stage_cfg['fuse_method']

        modules_list = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            # HighResolutionModule
            modules_list.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    fuse_method=fuse_method,
                    multi_scale_output=reset_multi_scale_output
                )
            )
            # change `num_inchannels` for next module.
            num_inchannels = modules_list[-1].get_num_inchannels()
        return nn.Sequential(*modules_list), num_inchannels

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        # transition1 and stage2
        x_list = []
        for i in range(self.stage_cfg_list[0].num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # transition2 and stage3
        # transition3 and stage4
        for j in range(1, 3):
            x_list = []
            for i in range(self.stage_cfg_list[j].num_branches):
                if eval('self.transition{}'.format(j + 1))[i] is not None:
                    x_list.append(eval('self.transition{}'.format(j + 1))[i](y_list[-1]))  # the last one
                else:
                    x_list.append(y_list[i])
            y_list = eval('self.stage{}'.format(j + 2))(x_list)

        # final layer
        x = self.final_layer(y_list[0])

        return x


def get_pose_net(cfg, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)
    is_train = cfg.keypoints_model.is_train
    if is_train and cfg.keypoints_model.init_weight:
        model.init_weight()

        if osp.isfile(cfg.keypoints_model.pretrained):
            pretrained_state_dict = torch.load(cfg.keypoints_model.pretrained)
            print('=> loading pretrained model {}'.format(cfg.keypoints_model.pretrained))

            # pretrained_state_dict ==> need_init_state_dict
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in model.pretrained_layers \
                        or model.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            model.load_state_dict(need_init_state_dict, strict=False)
        elif cfg.model.pretrained:
            print('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(cfg.keypoints_model.pretrained))
    else:
        model.load_state_dict(torch.load(cfg.keypoints_model.test.model_file))

    return model


if __name__ == '__main__':
    from MDRSREID.Settings.config.default_config import keypoints_model_cfg

    model = get_pose_net(cfg=keypoints_model_cfg,
                         is_train=False)

    print(model)
