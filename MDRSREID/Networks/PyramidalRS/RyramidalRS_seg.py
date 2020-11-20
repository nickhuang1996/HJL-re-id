import torch.nn as nn


class PyramidalRSSeg(nn.Module):
    """
    Use after ResNet backbone
    """
    def __init__(self, cfg):
        super(PyramidalRSSeg, self).__init__()
        self.cfg = cfg
        # output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
        # There:
        # output = (input - 1) * 2 + 1 - 2 * 1 + 3
        #        = 2 * input - 2 + 1 - 2 + 3
        #        = 2 * input
        # So it expand 2 times.
        self.deconv = nn.ConvTranspose2d(
            in_channels=cfg.model.seg.in_c,  # 2048
            out_channels=cfg.model.seg.out_c,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(cfg.model.seg.out_c)  # 256
        self.relu = nn.ReLU(inplace=True)
        # output = (input - kernel_size + 2 * padding) / stride + 1
        #        = (input - 1 + 2 * 0) / 1 + 1
        #        = input
        # So it keep unchanged
        self.conv = nn.Conv2d(
            in_channels=cfg.model.seg.out_c,  # 256
            out_channels=cfg.model.seg.num_classes,  # 8
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, in_dict):
        """
        [N, 2048, 24, 8] ==> [N, 256, 48, 16] ==> [N, 8, 48, 16]

        """
        feat = in_dict['feat']

        seg_pred = self.conv(self.relu(self.bn(self.deconv(feat))))

        return seg_pred

