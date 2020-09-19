import torch.nn as nn
import torch
from .AdaptiveDirectionGraphConvLayer import AdaptiveDirectionGraphConvLayer


class GraphConvNet(nn.Module):
    def __init__(self,
                 adj,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 scale):
        """
        :param adj:
        :param in_dim: 2048
        :param hidden_dim: 2048
        :param out_dim: 2048
        :param scale:
        This net includes 2 adgcn and 1 ReLU
        """
        super(GraphConvNet, self).__init__()

        self.adgcn1 = AdaptiveDirectionGraphConvLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            adj=adj,
            scale=scale
        )
        self.adgcn2 = AdaptiveDirectionGraphConvLayer(
            in_dim=hidden_dim,
            out_dim=out_dim,
            adj=adj,
            scale=scale
        )
        self.relu = nn.ReLU(inplace=True)

    def __call__(self, feat_vec_list):
        unsqueeze_feat_list = [feat.unsqueeze(1) for feat in feat_vec_list]  # 14 * [N, 2048] ==> 14 * [N, 1, 2048]
        cated_feat = torch.cat(unsqueeze_feat_list, dim=1)  # [N, 14, 2048]

        middle_feat = self.adgcn1(cated_feat)   # [N, 14, 2048]
        out_feat = self.adgcn2(middle_feat)   # [N, 14, 2048]

        gcned_feat_vec_list = []
        for i in range(out_feat.shape[1]):
            out_feat_i = out_feat[:, i].squeeze(1)  # [N, 2048]
            gcned_feat_vec_list.append(out_feat_i)

        return gcned_feat_vec_list  # 14 * [N, 2048]
