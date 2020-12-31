from __future__ import print_function
import os.path as osp
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import namedtuple
from MDRSREID.utils.load_state_dict import load_state_dict

from MDRSREID.Networks.RESIDUAL_NETWORK.BasicBlock import BasicBlock
from MDRSREID.Networks.RESIDUAL_NETWORK.Bottleneck import Bottleneck
from MDRSREID.Networks.RESIDUAL_NETWORK.convolution_layers import conv1x1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#

class ResNet(nn.Module):
    """
    I modify the original ResNet:
    For the variables:
        Remove the num_classes=1000
        Remove the avgpool,fc
        Define out_c = 512 * block.expansion

        Thus, the forward part is changed.
    For the function:
        The init methods is replaced by 'nn.init'

    For the layers:
        I create a new layer con1x1 to replace the original conv2d in '_make_layer'
    """

    def __init__(self, block, layers, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=cfg.model.backbone.last_conv_stride)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_c = 512 * block.expansion

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


# Use the namedtuple we can easily define the resnet.
ArchCfg = namedtuple('ArchCfg', ['block', 'layers'])
arch_dict = {
    'resnet18': ArchCfg(BasicBlock, [2, 2, 2, 2]),
    'resnet34': ArchCfg(BasicBlock, [3, 4, 6, 3]),
    'resnet50': ArchCfg(Bottleneck, [3, 4, 6, 3]),
    'resnet101': ArchCfg(Bottleneck, [3, 4, 23, 3]),
    'resnet152': ArchCfg(Bottleneck, [3, 8, 36, 3]),
}


def get_resnet(cfg):
    """download the ResNet model and return."""
    # Only ResNet layer4 use cfg, which is cfg.last_conv_stride
    model = ResNet(arch_dict[cfg.model.backbone.name].block, arch_dict[cfg.model.backbone.name].layers, cfg)
    # Determine the ResNet model if to be pre-trained or not.
    if cfg.model.backbone.pretrained:
        state_dict = model_zoo.load_url(url=model_urls[cfg.model.backbone.name],
                                        model_dir=cfg.model.backbone.pretrained_model_dir)
        load_state_dict(model, state_dict)
        # model_path = osp.abspath(osp.join(cfg.pretrained_model_dir, osp.basename(model_urls[cfg.name])))
        print('=> Loaded ImageNet Model: {}'.format(
            osp.join(cfg.model.backbone.pretrained_model_dir, osp.basename(model_urls[cfg.model.backbone.name]))))
    return model
