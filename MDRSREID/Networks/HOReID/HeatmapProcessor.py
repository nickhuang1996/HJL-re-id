import torch
import torch.nn as nn
import torch.nn.functional as F

# from MDRSREID.Settings.config.default_config import keypoints_model_cfg


class HeatmapProcessor2(object):
    def __init__(self,
                 cfg,
                 normalize_heatmap=True,
                 group_mode="sum",
                 norm_scale=1.0):
        self.cfg = cfg

        self.num_joints = cfg.keypoints_model.num_joints  # 17
        self.groups = cfg.keypoints_model.joints_groups

        self.group_mode = group_mode
        self.normalize_heatmap = normalize_heatmap
        self.norm_scale = norm_scale

        assert group_mode in ['sum', 'max'], "{} only support `sum` or `max`!".format(group_mode)

    def __call__(self, x):
        """
        :param x:
        :return:
            heatmap: after cat, input tensor
            max_response_2: after cat , indices
            max_index: after reorder, size is [n, c, 2]:
                        [:, :, 0] is column
                        [:, :, 1] is row
        """
        x = F.interpolate(x, size=[16, 8], mode='bilinear', align_corners=False)  # [N, 17, 64, 32] ==> [N, 17, 16, 8]
        n, c, h, w = x.shape

        # [n, c, hxw]
        x_reshaped = x.reshape((n, c, -1))  # [N, 17, 128]

        idx = torch.argmax(x_reshaped, dim=2)  # [N, 17]
        max_response, _ = torch.max(x_reshaped, dim=2)  # _ is equal to idx, [N, 17]

        idx = idx.reshape((n, c, 1))  # # [N, 17, 1]
        max_response = max_response.reshape((n, c)) # [N, 17]
        max_index = torch.empty((n, c, 2))  # [N, 17, 2]
        max_index[:, :, 0] = idx[:, :, 0] % w  # column 8
        max_index[:, :, 1] = idx[:, :, 0] // w  # row 16

        # cat max indices and values for 17 keypoints
        if self.group_mode == 'sum':
            heatmap = torch.sum(x[:, self.groups[0]], dim=1, keepdim=True)  # [N, 5, 16, 8] ==> [N, 1, 16, 8]
            max_response_2 = torch.mean(max_response[:, self.groups[0]], dim=1, keepdim=True)  # [N, 5] ==> [N, 1]

            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)  # [N, 12+1, 16, 8]

                max_response_i = torch.mean(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)  # [N, 12+1]

        elif self.group_mode == 'max':
            heatmap, _ = torch.max(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2, _ = torch.max(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi, _ = torch.max(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i, _ = torch.max(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)
        else:
            raise ValueError("{} is not supported! It should be `sum` or `max`.".format(self.group_mode))

        if self.normalize_heatmap:
            heatmap = self.normalize(in_tensor=heatmap,
                                     norm_scale=self.norm_scale)

        return heatmap, max_response_2, max_index  # [N, 12+1, 16, 8] [N, 12+1] [N, 17, 2]

    def normalize(self, in_tensor, norm_scale):
        """
        This function use `softmax` to normalize
        :param in_tensor:  [N, 13, 16, 8]
        :param norm_scale: default as `1.0`
        :return:
        """
        n, c, h, w = in_tensor.shape  # [N, 13, 16, 8]
        in_tensor_reshape = in_tensor.reshape((n, c, -1))  # [N, 13, 128]

        normalized_tensor = F.softmax(norm_scale * in_tensor_reshape, dim=2)  # [N, 13, 128]
        normalized_tensor = normalized_tensor.reshape((n, c, h, w))

        return normalized_tensor
