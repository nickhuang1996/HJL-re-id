import torch.nn as nn


class HighResolutionModule(nn.Module):
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 fuse_method,
                 multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self._check_branches(num_branches,
                             num_blocks,
                             num_inchannels,
                             num_channels)

        self.num_inchannels = num_channels
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output  # This is used in `_make_fuse_layers`

        self.branches = self._make_branches(
            num_branches=num_branches,
            block=blocks,
            num_blocks=num_blocks,
            num_channels=num_channels
        )

        self.fuse_layers = self._make_fuse_layers()  # nn.ModuleList()
        self.relu = nn.ReLU(True)

    def _check_branches(self,
                        num_branches,
                        num_blocks,
                        num_inchannels,
                        num_channels):
        """
        :param num_branches:
        :param num_blocks:
        :param num_inchannels:
        :param num_channels:
        :return:
        This function checks as below:
        `num_branches` should be `equal` to the length of
            `num_blocks`
            `num_channels`
            `num_inchannels`
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_branches(self,
                       num_branches,
                       block,
                       num_blocks,
                       num_channels):
        branches_list = []

        for i in range(num_branches):
            branches_list.append(
                self._make_one_branch(branch_index=i,
                                      block=block,
                                      num_blocks=num_blocks,
                                      num_channels=num_channels,
                                      )    # stride = 1
            )
        return nn.ModuleList(branches_list)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """
        :param branch_index:
        :param block:
        :param num_blocks:
        :param num_channels:
        :param stride:
        :return:
        This function is similar to ResNet's `_make_layer`
        """
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """
        :return: fuse_layers_list

        If num_branches is 1, do not fuse layers.
        if j > i:
            feature maps' size is not changed, including upsample.
        else:
            feature maps' size is half step by step, downsample.
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers_list = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer_list = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(  # size is not changed.
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')  # Upsample scale is `2 ^ (j - i)`
                        )
                    )
                elif j == i:
                    fuse_layer_list.append(None)
                else:
                    Conv3x3s_list = []
                    for k in range(i - j):
                        if k == i - j - 1:  # the last Conv3x3, no ReLU
                            num_outchannels_Conv3x3 = num_inchannels[i]
                            Conv3x3s_list.append(
                                nn.Sequential(  # size is 0.5 size
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_Conv3x3,  # i
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_Conv3x3)
                                )
                            )
                        else:  # not the last one
                            num_outchannels_Conv3x3 = num_inchannels[j]
                            Conv3x3s_list.append(
                                nn.Sequential(
                                    nn.Conv2d(  # size is 0.5 size
                                        num_inchannels[j],
                                        num_outchannels_Conv3x3,  # j
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_Conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer_list.append(nn.Sequential(*Conv3x3s_list))
            fuse_layers_list.append(nn.ModuleList(fuse_layer_list))  # Only one element?
        return nn.ModuleList(fuse_layers_list)

    def get_num_inchannels(self):
        """
        :return: num_inchannels
        This function is used in PoseHighResolutionNet `_make_stage`

        **Note**:
        `self.num_inchannels` is changed in `_make_one_branch`:
            self.num_inchannels[branch_index] = \
                num_channels[branch_index] * block.expansion
        """
        return self.num_inchannels

    def forward(self, x):
        """
        :param x:
        :return:

        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
