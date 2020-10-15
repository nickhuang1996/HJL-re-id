import torch.nn as nn


class MDRSMultiSeg(nn.Module):
    """
    Use after ResNet backbone
    """
    def __init__(self, cfg):
        super(MDRSMultiSeg, self).__init__()
        self.cfg = cfg
        # output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
        # There:
        # output = (input - 1) * 2 + 1 - 2 * 1 + 3
        #        = 2 * input - 2 + 1 - 2 + 3
        #        = 2 * input
        # So it expand 2 times.
        self.deconv = nn.ConvTranspose2d(
            in_channels=cfg.model.multi_seg.in_c,  # 2048
            out_channels=cfg.model.multi_seg.mid_c,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        # output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
        # There:
        # output = (input - 1) * 3 + 2 - 2 * 1 + 3
        #        = 3 * input - 3 + 2 - 2 + 3
        #        = 3 * input
        # So it expand 3 times.
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=cfg.model.multi_seg.in_c1,  # 2048
            out_channels=cfg.model.multi_seg.in_c2,  # 1024
            kernel_size=3,
            stride=3,
            padding=1,
            output_padding=2,
            bias=False,
        )
        self.deconv3_pool = nn.AdaptiveAvgPool2d((24, 8))
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=cfg.model.multi_seg.in_c1,  # 2048
            out_channels=cfg.model.multi_seg.in_c2,  # 1024
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=cfg.model.multi_seg.in_c2,  # 1024
            out_channels=cfg.model.multi_seg.mid_c,  # 256
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(cfg.model.multi_seg.mid_c)  # 256
        self.relu = nn.ReLU(inplace=True)
        # output = (input - kernel_size + 2 * padding) / stride + 1
        #        = (input - 1 + 2 * 0) / 1 + 1
        #        = input
        # So it keep unchanged
        self.conv = nn.Conv2d(
            in_channels=cfg.model.multi_seg.mid_c,  # 256
            out_channels=cfg.model.multi_seg.num_classes,  # 8
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, in_dict):
        """
        Option 1:
        [N, 2048, 8, 3] ==>  [N, 1024, 24, 9] ==> [N, 1024, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]
        [N, 2048, 12, 4] ==>  [N, 1024, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]
        [N, 2048, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]

        Option 2:
        [N, 2048, 12, 4] ==>  [N, 1024, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]
        [N, 2048, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]
        [N, 2048, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]

        """
        feat_list = in_dict['feat_list']
        multi_seg_pred_list = []
        # Option 1
        # for i in range(self.cfg.blocksetting.multi_seg):
        #     if i == 0:
        #         multi_seg_pred_list.append(self.conv(self.relu(self.bn(self.deconv2(self.deconv3_pool(self.deconv3(feat_list[i])))))))
        #     elif i == 1:
        #         multi_seg_pred_list.append(self.conv(self.relu(self.bn(self.deconv2(self.deconv1(feat_list[i]))))))
        #     elif i == 2:
        #         multi_seg_pred_list.append(self.conv(self.relu(self.bn(self.deconv(feat_list[i])))))
        # Option 2
        for i in range(self.cfg.blocksetting.multi_seg):
            if i == 0:
                multi_seg_pred_list.append(self.conv(self.relu(self.bn(self.deconv2(self.deconv1(feat_list[i]))))))
            else:
                multi_seg_pred_list.append(self.conv(self.relu(self.bn(self.deconv(feat_list[i])))))

        return multi_seg_pred_list

